from transformers.models.llama.configuration_llama import LlamaConfig
            

class LlamaCLConfig(LlamaConfig):
    def __init__(self, vocab_size=32000, hidden_size=4096, intermediate_size=11008, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=None, hidden_act="silu", max_position_embeddings=2048, initializer_range=0.02, rms_norm_eps=0.000001, use_cache=True, pad_token_id=0, bos_token_id=1, eos_token_id=2, pretraining_tp=1, tie_word_embeddings=False, rope_scaling=None, **kwargs):
        super().__init__(vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, hidden_act, max_position_embeddings, initializer_range, rms_norm_eps, use_cache, pad_token_id, bos_token_id, eos_token_id, pretraining_tp, tie_word_embeddings, rope_scaling, **kwargs)

        self.model_name = 'meta-llama/Llama-2-7b-chat-hf'
        self.retriever_state_dict = None  # "/data/bhpeng/SLM-llama/vectordb/output/model_6_6_medical.pth" 
        self.disable_task_id = False

        self.pool_size = 12
        self.weight_topk = 2
        self.random_dropout = None
        self.low_rank = 8
        self.groups = 6
        self.similarity_type = 'cosine'
        self.pool_train_keys = False
        self.pool_train_weights = True
        self.task_pool_index_range = {
            "finance": [0, 4],
            "history": [4, 8],
            "medical": [8, 12]
        }

        config = LlamaConfig.from_pretrained(self.model_name)
        for k, v in config.__dict__.items():
            setattr(self, k, v)

