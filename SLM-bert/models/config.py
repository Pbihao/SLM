from transformers.models.bert.configuration_bert import BertConfig

class MultiHeadConfig(BertConfig):
    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, position_embedding_type="absolute", use_cache=True, classifier_dropout=None, **kwargs):
        super().__init__(vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, max_position_embeddings, type_vocab_size, initializer_range, layer_norm_eps, pad_token_id, position_embedding_type, use_cache, classifier_dropout, **kwargs)

        self.task_to_num_label = {
            'yelp': 5,
            'ag_news': 4,
            'yahoo': 10,
            'dbpedia': 14,
            'amazon': 5,
        }
        self.task_list = ['ag_news', 'amazon', 'dbpedia', "yahoo", "yelp"]

        self.disable_task_id = True
        self.adopt_pool_prefix_mask = False
        self.model_name = 'bert-base-uncased'

        self.retriever_state_dict = "/dataset/zhuotaotian/bhpeng/SLM-weight-ab/vectordb/output/model_6*20_all.pth"  # "/mnt/home/bhpeng22/research/llm/slm/vectordb/output/model_6*40_all.pth"
        self.pool_size = 20
        self.weight_topk = 4
        self.random_dropout = None
        self.low_rank = 12
        self.groups = 6
        self.similarity_type = 'cosine'
        self.pool_train_keys = False
        self.pool_train_weights = True
        self.task_pool_index_range = {
            "ag_news": [0, 4],
            "amazon": [4, 8],
            "dbpedia": [8, 12],
            "yahoo": [12, 16],
            "yelp": [16, 20]
        }

        

        
            

