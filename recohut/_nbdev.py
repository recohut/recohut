# AUTOGENERATED BY NBDEV! DO NOT EDIT!

__all__ = ["index", "modules", "custom_doc_links", "git_url"]

index = {"AOTMDataset": "aotm.ipynb",
         "download": "movielens.ipynb",
         "unzip": "movielens.ipynb",
         "AbstractDataset": "base.ipynb",
         "AbstractDatasetv2": "base.ipynb",
         "to_list": "retailrocket.ipynb",
         "files_exist": "retailrocket.ipynb",
         "Dataset": "retailrocket.ipynb",
         "SessionDataset": "base.ipynb",
         "SessionDatasetv2": "base.ipynb",
         "data_masks": "session.ipynb",
         "GraphData": "session.ipynb",
         "ML1MDataset": "movielens.ipynb",
         "Music30Dataset": "music30.ipynb",
         "NowPlayingDataset": "nowplaying.ipynb",
         "RetailRocketDatasetv2": "retailrocket.ipynb",
         "SampleSessionDataset": "session.ipynb",
         "split_validation": "session.ipynb",
         "Synthetic": "synthetic.ipynb",
         "Session": "synthetic.ipynb",
         "SequentialMarkov": "synthetic.ipynb",
         "LightGConv": "message_passing.ipynb",
         "LRGCCF": "message_passing.ipynb",
         "OUNoise": "ou_noise.ipynb",
         "calculate_precision_recall": "utils.ipynb",
         "calculate_ndcg": "utils.ipynb",
         "recall": "utils.ipynb",
         "ndcg": "utils.ipynb",
         "recalls_and_ndcgs_for_ks": "utils.ipynb",
         "Actor": "actor_critic.ipynb",
         "Critic": "actor_critic.ipynb",
         "BetaBandit": "bandits.ipynb",
         "EpsilonBandit": "bandits.ipynb",
         "Multi_Layer_Perceptron": "dnn.ipynb",
         "CollabFNet": "dnn.ipynb",
         "EmbeddingNet": "embedding.ipynb",
         "get_list": "embedding.ipynb",
         "GroupEmbedding": "embedding.ipynb",
         "EpsilonGreedy": "epsilon.ipynb",
         "EpsilonGreedyRunner": "epsilon.ipynb",
         "MF": "factorization.ipynb",
         "BiasedMF": "factorization.ipynb",
         "NeuMF": "factorization.ipynb",
         "GMF": "factorization.ipynb",
         "SiReN": "siren.ipynb",
         "SelfAttention_Layer": "attrec.ipynb",
         "AttRec": "attrec.ipynb",
         "BPR": "bpr.ipynb",
         "Caser": "caser.ipynb",
         "CrossNetwork": "dcn.ipynb",
         "DNN": "widedeep.ipynb",
         "DCN": "dcn.ipynb",
         "Residual_Units": "deepcross.ipynb",
         "DeepCross": "deepcross.ipynb",
         "DeepMF": "deepmf.ipynb",
         "FM_Layer": "fm.ipynb",
         "FM_Layer_v2": "fm.ipynb",
         "FM": "fm.ipynb",
         "FFM_Layer": "fm.ipynb",
         "FFM": "fm.ipynb",
         "NFM": "fm.ipynb",
         "AFM": "fm.ipynb",
         "DeepFM": "fm.ipynb",
         "DNN_v2": "fm.ipynb",
         "Linear": "widedeep.ipynb",
         "CIN": "fm.ipynb",
         "xDeepFM": "fm.ipynb",
         "NCF": "ncf.ipynb",
         "PNN": "pnn.ipynb",
         "get_angles": "sasrec.ipynb",
         "positional_encoding": "sasrec.ipynb",
         "scaled_dot_product_attention": "sasrec.ipynb",
         "MultiHeadAttention": "sasrec.ipynb",
         "FFN": "sasrec.ipynb",
         "EncoderLayer": "sasrec.ipynb",
         "SASRec": "sasrec.ipynb",
         "VDeepMF": "vdeepmf.ipynb",
         "VNCF": "vncf.ipynb",
         "WideDeep": "widedeep.ipynb",
         "seed_everything": "utils.ipynb",
         "DDPGAgent": "ddpg.ipynb",
         "Env": "recsys.ipynb",
         "ReplayMemory": "memory.ipynb",
         "simple_aggregate": "aggregate.ipynb",
         "BipartiteDataset": "bipartite.ipynb",
         "sparseFeature": "movielens.ipynb",
         "denseFeature": "criteo.ipynb",
         "create_criteo_dataset": "criteo.ipynb",
         "create_ml_1m_dataset": "movielens.ipynb",
         "create_implicit_ml_1m_dataset": "movielens.ipynb",
         "label_encode": "encode.ipynb",
         "simple_normalize": "normalize.ipynb",
         "simple_negative_sampling": "sampling.ipynb",
         "AbstractNegativeSampler": "sampling.ipynb",
         "RandomNegativeSampler": "sampling.ipynb",
         "PopularNegativeSampler": "sampling.ipynb",
         "GroupGenerator": "user_grouping.ipynb",
         "makedirs": "common_utils.ipynb",
         "wget_download": "common_utils.ipynb",
         "download_url": "common_utils.ipynb",
         "maybe_log": "common_utils.ipynb",
         "extract_tar": "common_utils.ipynb",
         "extract_zip": "common_utils.ipynb",
         "extract_bz2": "common_utils.ipynb",
         "extract_gz": "common_utils.ipynb",
         "Configurator": "config.ipynb",
         "set_logger": "logging.ipynb"}

modules = ["datasets/aotm.py",
           "datasets/base.py",
           "datasets/movielens.py",
           "datasets/music30.py",
           "datasets/nowplaying.py",
           "datasets/retailrocket.py",
           "datasets/session.py",
           "datasets/synthetic.py",
           "layers/message_passing.py",
           "layers/ou_noise.py",
           "metrics/utils.py",
           "models/actor_critic.py",
           "models/bandits.py",
           "models/dnn.py",
           "models/embedding.py",
           "models/epsilon.py",
           "models/factorization.py",
           "models/siren.py",
           "models/tf/attrec.py",
           "models/tf/bpr.py",
           "models/tf/caser.py",
           "models/tf/dcn.py",
           "models/tf/deepcross.py",
           "models/tf/deepmf.py",
           "models/tf/fm.py",
           "models/tf/ncf.py",
           "models/tf/pnn.py",
           "models/tf/sasrec.py",
           "models/tf/vdeepmf.py",
           "models/tf/vncf.py",
           "models/tf/widedeep.py",
           "pytorch/utils.py",
           "rl/agents/ddpg.py",
           "rl/envs/recsys.py",
           "rl/memory.py",
           "transforms/aggregate.py",
           "transforms/bipartite.py",
           "transforms/datasets/criteo.py",
           "transforms/datasets/movielens.py",
           "transforms/encode.py",
           "transforms/normalize.py",
           "transforms/sampling.py",
           "transforms/user_grouping.py",
           "utils/common_utils.py",
           "utils/config.py",
           "utils/logging.py"]

doc_url = "https://RecoHut-Projects.github.io/recohut/"

git_url = "https://github.com/RecoHut-Projects/recohut/tree/master/"

def custom_doc_links(name): return None
