import sys
#sys.path.append('/content/drive/MyDrive/Colab Notebooks/transformer')
#sys.path.append('/content/drive/MyDrive/Colab Notebooks/transformer/other')
from embedding.TransformerEmbedding import TransformerEmbedding
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.Transformer import Transformer
from other.dataloader import DataLoaderHF
from other.dataloader_wmt import DataLoaderWMT
from other.BLEU import bleu_stats, bleu, get_bleu
import torch
import torch.nn as nn
import time
from config import Config
from train import *
from test import evaluate, inference

# 临时关闭代理
import os
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
    os.environ.pop(k, None)

# 初始化数据
#dataloader = DataLoaderHF(Config.model_name, Config.max_len, Config.batch_size, Config.special_token)
dataloader = DataLoaderWMT(Config.model_name, Config.max_len, Config.batch_size, Config.special_token) #wmt数据集
tokenizer = dataloader.tokenizer
tokenizer.add_special_tokens({'additional_special_tokens':[Config.special_token]})
voc_size = len(tokenizer.get_vocab())
pad_id = tokenizer.pad_token_id
train_data, valid_data, test_data = dataloader.make_dataset()
train_iter, valid_iter, test_iter = dataloader.make_iter(train_data, valid_data, test_data)
step_per_epoch = len(train_iter)
total_steps = step_per_epoch * Config.epoches
warmup_steps = int(total_steps * 0.015)
print(total_steps, warmup_steps)

# 建立模型
model = Transformer(pad_idx=pad_id,
                    enc_voc_size=voc_size,
                    dec_voc_size=voc_size,
                    d_model=Config.d_model,
                    max_len=Config.max_len,
                    batch_size=Config.batch_size,
                    n_head=Config.n_head,
                    n_layers=Config.n_layers,
                    ffn_hidden=Config.ffn_hidden,
                    drop_prob=Config.drop_prob,
                    device=Config.device)

model.to(Config.device)
model = torch.compile(model)
print('model has {0} parameters'.format(count_parameters(model)))
model.apply(initialize_weights)

# 创建优化器和调度器
optimizer = create_optimizer(model, Config.init_lr, Config.weight_decay, Config.adam_eps)
print(Config.init_lr)
scheduler = create_scheduler(optimizer, warmup_steps, total_steps)
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

# 运行函数
def run(total_epoch, best_loss):
    # 初始化日志记录器
    logger = TrainerLogger()
    
    train_losses, test_losses, bleus = [], [], []
    for epoch in range(total_epoch):
        start_time = time.time()
        
        # 训练并获取损失
        train_loss = train(model, train_iter, optimizer, criterion, scheduler, Config.clip, logger)
        valid_loss, bleu_score = evaluate(model, valid_iter, criterion, tokenizer)
        
        # 计算epoch时间
        end_time = time.time()
        epoch_secs = end_time - start_time
        
        # 记录日志
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=valid_loss,
            epoch_time=epoch_secs,
            bleu_score=bleu_score
        )
        
        print('Epoch: {0}'.format(epoch))
        print('Train Loss: {0:.3f} | Val Loss: {1:.3f} | BLEU: {2:.3f}'.format(
            train_loss, valid_loss, bleu_score))
        print('################################################################')
        
        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu_score)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    # 关闭日志记录器
    logger.close()
    return train_losses, test_losses, bleus

if __name__ == '__main__':
    
    start_time = time.time()
    train_losses, test_losses, bleus = run(Config.epoches, Config.inf)
    end_time = time.time()

    mins, secs = epoch_time(start_time, end_time)
    print(f"Training completed in {mins}m {secs}s")
    
    #测试模型
    test_bleu = inference(model, test_iter, tokenizer, pad_id)
    print(f"Test BLEU: {sum(test_bleu) / len(test_bleu):.3f}")
    
    # 示例推理
    batch = next(iter(test_iter))
    a = batch['input_ids'].to(Config.device)
    b = batch['labels'].to(Config.device)

    decoder_output = torch.full((Config.batch_size, Config.max_len), pad_id)
    decoder_output[:, 0] = 58101

    src_mask = model.make_src_mask(a)
    model.eval()
    encoder_output = model.encoder(a, src_mask)
    for j in range(1, Config.max_len):
        trg_mask = model.make_trg_mask(decoder_output)
        output = model.decoder(encoder_output, decoder_output, src_mask, trg_mask)
        output = output.argmax(dim=2)
        output = output[:, j-1]
        decoder_output[:, j] = output

    for i in range(5):
        print("真实数据：{0}".format(tokenizer.decode(b[i], skip_special_tokens=True)))
        print("模型数据：{0}".format(tokenizer.decode(decoder_output[i], skip_special_tokens=True)))
        bleu_score = get_bleu(hypothesis=tokenizer.decode(b[i], skip_special_tokens=True).split(), 
                             reference=tokenizer.decode(decoder_output[i], skip_special_tokens=True).split())
        print(f"BLEU: {bleu_score:.3f}")
        print('################################################################')