

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Any, Mapping, Union, List, Sequence
from transformers.tokenization_utils_base import BatchEncoding
from einops import rearrange
import math
import warnings
from copy import deepcopy



class Config(object):
    def __init__(self,
                 pool_size=8,
                 random_dropout = 0.25,
                 weight_topk=2,
                 groups = 6,
                 hidden_size=4096,
                 num_hidden_layers = 32,
                 similarity_type = 'cosine',
                 pool_train_keys = True,
                 pool_train_weights = False) -> None:
        self.pool_size =pool_size
        self.random_dropout = random_dropout
        self.weight_topk = weight_topk
        self.groups = groups
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.similarity_type = similarity_type
        self.pool_train_keys = pool_train_keys
        self.pool_train_weights = pool_train_weights


def generate_orthogonal_matrix(rows, cols):
    tensor = torch.empty(rows, cols)
    indexes = list(range(0, rows, cols))
    if cols not in indexes:
        indexes.append(cols)
    for i in range(len(indexes) - 1):
        nn.init.orthogonal_(tensor[indexes[i]: indexes[i+1], :])
    return tensor


class Retriever(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        assert config.similarity_type in ['cosine', 'softmax'], "The similarity calculation should be ['cosine', softmax]"
        self.similarity_type = config.similarity_type
        self.pool_size = config.pool_size
        self.weight_topk = config.weight_topk 
        self.hidden_size = config.hidden_size
        self.groups = config.groups
        self.random_dropout = config.random_dropout
        
        self.pool_train_keys = config.pool_train_keys
        self.pool_train_weights = config.pool_train_weights
        self.num_hidden_layers = config.num_hidden_layers
        self.low_rank = config.low_rank

        if self.random_dropout is not None:
            assert self.random_dropout < 1 and self.random_dropout >=0, "random_dropout should be in [0, 1)"        

        # self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/nli-roberta-base-v2')
        # self.bert = AutoModel.from_pretrained('sentence-transformers/nli-roberta-base-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/nli-mpnet-base-v2')
        self.bert = AutoModel.from_pretrained('sentence-transformers/nli-mpnet-base-v2')

        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.model_hidden_size = self.bert.get_input_embeddings().weight.shape[-1]
        assert self.model_hidden_size % self.groups == 0, "The vectordb hidden size: {} can not be divided envenly by the groups:{}".format(self.model_hidden_size, self.groups)
        self.key_hidden_size = self.model_hidden_size // self.groups

        # weight offset 
        low_rank_a = torch.zeros(self.pool_size, self.hidden_size * self.config.low_rank * self.num_hidden_layers)
        low_rank_b = nn.init.normal_(torch.empty(self.pool_size,  self.hidden_size * self.config.low_rank * self.num_hidden_layers))
        self.weight_offset = nn.parameter.Parameter(torch.stack([low_rank_a, low_rank_b], dim=-2))  # [pool_size, 2, channels*l*r]

        if self.pool_size > self.key_hidden_size:
            warnings.warn("The pool size is larger than the key_hidden_size, may cause the generate unstable keys")
        keys = [generate_orthogonal_matrix(self.pool_size, self.key_hidden_size) for _ in range(self.groups)]
        keys = torch.stack(keys, dim=0).unsqueeze(0)  # [1, groups, pool_size, key_hidden_size]
        self.keys = nn.parameter.Parameter(keys)

        self.last_keys = None  # Used for the centrifugal loss calculation

        if not config.pool_train_keys:
            self.freeze_keys()
        if not config.pool_train_weights:
            self.freeze_weights()

    # =============================================================================================
    # For the retriever process, we follow and modify from:
    # https://github.com/google-research/l2p [Learning to Prompt for Continual Learning (L2P)]
    # Thanks for their open-work
    # =============================================================================================
    def forward(
            self, 
            inputs, 
            pool_mask=None,
            return_topk_index=False,
            use_distance_weight=True):
        
        queries = self.encode(inputs)
        bsz = queries.shape[0]
        queries = rearrange(queries, "b (g c) -> b g c", g=self.groups)
        keys = self.keys.repeat(bsz, 1, 1, 1)
        outputs = dict()

        if self.similarity_type == 'cosine':
            queries = queries.unsqueeze(2).repeat(1, 1, self.pool_size, 1)
            sim = F.cosine_similarity(queries, keys, dim=-1)  # [bsz, groups, pool_size]
        else:
            queries = queries.unsqueeze(2).repeat(1, 1, self.pool_size, 1)
            sim = F.cosine_similarity(queries, keys, dim=-1) / 0.1  # [bsz, groups, pool_size]
            sim = torch.softmax(sim, dim=-1)

        idx_sim = sim.clone().detach()
        if self.training and self.random_dropout is not None:
            idx = torch.rand_like(idx_sim) <= self.random_dropout
            idx_sim.masked_fill_(idx, -100.)
        if pool_mask is not None:
            idx_sim[:, :, pool_mask == 0] = -100.
        
        if not use_distance_weight:
            _, idx = idx_sim.topk(self.weight_topk, dim=-1)  # [bsz, group, topk]
            idx_vote = rearrange(idx, "b g k -> g (b k)")
            base = (torch.arange(0, self.groups, device=idx_vote.device) * self.pool_size).view(-1, 1)
            idx_vote = (base + idx_vote).flatten()
            bin_count = torch.bincount(idx_vote, minlength=self.pool_size*self.groups).view(self.groups, self.pool_size)
            idx_vote = torch.topk(bin_count, k=self.weight_topk)[1]  # [groups, topk]
        else:
            idx_sim = torch.mean(idx_sim, dim=[0,1])
            dis_weihgt, idx_vote = idx_sim.topk(self.weight_topk, dim=-1) # [topk]
            dis_weihgt = dis_weihgt / (dis_weihgt.sum() + 1e-9)

        weight_offset = torch.take_along_dim(self.weight_offset, idx_vote[:, None,None], dim=0)

        low_rank_a = weight_offset[..., 0,:].view(self.weight_topk, self.num_hidden_layers, self.low_rank, self.hidden_size)
        low_rank_b = weight_offset[..., 1,:].view(self.weight_topk, self.num_hidden_layers, self.low_rank, self.hidden_size)

        weight_offset = torch.einsum("n l r x, n l r y -> n l x y", low_rank_a, low_rank_b)
        
        if not use_distance_weight:
            weight_offset = torch.mean(weight_offset, dim=0)
        else:
            weight_offset = (dis_weihgt[:, None, None, None] * weight_offset).sum(0)


        outputs['weight_offset'] = weight_offset

        if self.pool_train_keys:
            sim = torch.take_along_dim(sim, idx, dim=-1)
            loss = -sim.mean()
            outputs['key_loss'] = loss

        if return_topk_index:
            outputs['topk_index'] = idx
        
        return outputs
    
    def encode(self, inps):
        with torch.no_grad():
            if isinstance(inps, str):
                inps = [inps]
            if isinstance(inps, Sequence):
                inps = self.tokenizer(inps, padding=True, truncation=True, return_tensors='pt')
            assert isinstance(inps, BatchEncoding), "The inputs of the encoder should be BatchEncoding."
            inps = inps.to(self.bert.device)
            embeddings = self.bert(**inps)
            embeddings = self.mean_pooling(embeddings, inps['attention_mask'])
        return embeddings
    
    def tokenize(self, sentences: Union[str, List[str]]):
        if isinstance(sentences, str):
            sentences = [sentences]
        return self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def freeze_keys(self):
        print("<=============== Freeze Keys =============>")
        self.keys.requires_grad = False
    
    def freeze_weights(self):
        print("<=============== Freeze weights =============>")
        self.weight_offset.requires_grad = False
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_keys(self, values):
        assert values.shape == torch.Size([1, self.groups, self.pool_size, self.key_hidden_size]), "The shape of values: {} don't equal to {}".format(values.shape, [1, self.groups, self.pool_size, self.key_hidden_size])
        self.keys = nn.parameter.Parameter(values)
        if not self.pool_train_keys:
            self.freeze_keys()

    def set_last_keys(self, values):
        assert values.size(-1) == self.key_hidden_size and values.size(1) == self.groups, "The last keys shape: {} doesn't fit the retriever shape: {}".format(values.shape, self.keys.shape)
        self.last_keys = values
