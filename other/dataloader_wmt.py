import pyarrow.parquet as pq
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import os
import random

class WMT14Dataset(Dataset):
    def __init__(self, file_paths, tokenizer, max_len, sos_token):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sos_token = sos_token
        self.data = []
        
        # 读取所有Parquet文件
        for file_path in file_paths:
            table = pq.read_table(file_path)
            self.data.extend(table.to_pylist())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]['translation']
        source_text = item['de']
        target_text = f"{self.sos_token}{item['en']}"
        
        # 编码源文本和目标文本
        source = self.tokenizer(
            source_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target = self.tokenizer(
            target_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source['input_ids'].squeeze(0),
            'labels': target['input_ids'].squeeze(0)
        }

class DataLoaderWMT:
    def __init__(self, model_name, max_len, batch_size, sos_token):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_len = max_len
        self.sos_token = sos_token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 定义文件路径
        self.data_dir = "/root/autodl-tmp/datasets/wmt14/de-en"
        self.train_files = [
            os.path.join(self.data_dir, "train-00000-of-00003.parquet"),
            os.path.join(self.data_dir, "train-00001-of-00003.parquet"),
            os.path.join(self.data_dir, "train-00002-of-00003.parquet")
        ]
        self.valid_file = os.path.join(self.data_dir, "validation-00000-of-00001.parquet")
        self.test_file = os.path.join(self.data_dir, "test-00000-of-00001.parquet")

    def make_dataset(self):
        print("Loading datasets...")
        train_data = WMT14Dataset(self.train_files, self.tokenizer, self.max_len, self.sos_token)
        valid_data = WMT14Dataset([self.valid_file], self.tokenizer, self.max_len, self.sos_token)
        test_data = WMT14Dataset([self.test_file], self.tokenizer, self.max_len, self.sos_token)
        print("Datasets loaded successfully!")
        return train_data, valid_data, test_data

    def make_iter(self, train_data, valid_data, test_data):
        # 创建训练集DataLoader（完整数据）
        train_iter = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=False
        )
        
        # 创建验证集和测试集的子集（20%数据）
        valid_size = int(0.2 * len(valid_data))
        test_size = int(0.2 * len(test_data))
        
        # 随机选择20%的索引
        valid_indices = random.sample(range(len(valid_data)), valid_size)
        test_indices = random.sample(range(len(test_data)), test_size)
        
        valid_subset = Subset(valid_data, valid_indices)
        test_subset = Subset(test_data, test_indices)
        
        valid_iter = DataLoader(
            valid_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False
        )
        
        test_iter = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False
        )
        
        return train_iter, valid_iter, test_iter