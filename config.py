import torch


class Config:
    # 模型参数
    d_model = 512
    max_len = 256
    batch_size = 50
    n_head = 8
    n_layers = 6
    ffn_hidden = 2048
    drop_prob = 0.1

    # 优化器参数
    init_lr = 5e-4
    factor = 0.9
    adam_eps = 5e-9
    patience = 10
    warmup = 20
    epoches = 100
    clip = 1.0
    weight_decay = 5e-4
    inf = float('inf')

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据配置
    model_name = '/root/autodl-tmp/models/opus-mt-de-en'
    special_token = "<s>"