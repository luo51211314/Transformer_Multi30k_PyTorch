import torch
import torch.nn as nn
import time
import gc
from tqdm import tqdm
from config import Config
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from logger import TrainerLogger

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)

def create_optimizer(model, init_lr, weight_decay, adam_eps):
    return AdamW(params=model.parameters(),
                lr=init_lr,
                weight_decay=weight_decay,
                eps=adam_eps)

def create_scheduler(optimizer, warmup_steps, total_steps):
    return get_cosine_schedule_with_warmup(optimizer, 
                                         num_warmup_steps=warmup_steps,
                                         num_training_steps=total_steps)

def train(model, iterator, optimizer, criterion, scheduler, clip, logger=None):
    model.train()
    epoch_loss = 0
    scaler = torch.amp.GradScaler(device='cuda')
    
    # 使用tqdm添加进度条
    progress_bar = tqdm(
        enumerate(iterator), 
        total=len(iterator),
        desc=f"Training",
        leave=False,
        dynamic_ncols=True
    )

    for i, batch in progress_bar:
        src = batch['input_ids'].to(Config.device, non_blocking=True)
        trg = batch['labels'].to(Config.device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = model(src, trg)
            output_reshape = output[:, :-1, :]
            output_reshape = output_reshape.contiguous().view(-1, output_reshape.shape[-1]).to(Config.device)
            trg = trg[:, 1:]
            trg = trg.contiguous().view(-1).to(Config.device)
            loss = criterion(output_reshape, trg)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        epoch_loss += loss.item()
        
        # 更新进度条描述
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })

    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - elapsed_mins * 60)
    return elapsed_mins, elapsed_secs