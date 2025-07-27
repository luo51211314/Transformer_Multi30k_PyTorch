import logging
import torch
import time
from logging.handlers import RotatingFileHandler

class TrainerLogger:
    def __init__(self, log_file='training.log'):
        self.logger = logging.getLogger('TransformerTrainer')
        self.logger.setLevel(logging.INFO)
        
        # 防止重复添加handler
        if not self.logger.handlers:
            # 文件handler - 滚动日志(最大1MB，保留3个备份)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=1024 * 1024, backupCount=3, encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)
            
            # 控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 日志格式
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def log_epoch(self, epoch, train_loss, val_loss, epoch_time, gpu_mem=None):
        # 获取内存使用情况
        cpu_mem = f"{torch.cuda.memory_allocated()/1024**2:.2f}MB" if torch.cuda.is_available() else "N/A"
        
        # 如果有GPU则获取GPU内存
        gpu_info = ""
        if torch.cuda.is_available():
            gpu_mem_alloc = torch.cuda.memory_allocated()/1024**2
            gpu_mem_cached = torch.cuda.memory_reserved()/1024**2
            gpu_info = f" | GPU alloc: {gpu_mem_alloc:.2f}MB | GPU cache: {gpu_mem_cached:.2f}MB"
        
        # 记录日志
        self.logger.info(
            f"Epoch {epoch} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Time: {epoch_time:.2f}s - "
            f"CPU Mem: {cpu_mem}"
            f"{gpu_info}"
        )
    
    def close(self):
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)