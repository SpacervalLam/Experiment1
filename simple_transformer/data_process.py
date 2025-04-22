import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import AutoTokenizer



class TranslationDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128, is_train=True, test_size=0.1):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.test_size = test_size
        self._load_data_info()

    def _load_data_info(self):
        # 先统计总行数用于分割训练/验证集
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.total_lines = sum(1 for _ in f)
        
        # 计算训练集和验证集的分割点
        train_size = int((1 - self.test_size) * self.total_lines)
        if self.is_train:
            self.start_idx = 0
            self.end_idx = train_size
        else:
            self.start_idx = train_size
            self.end_idx = self.total_lines

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        # 计算实际文件中的行号
        actual_idx = self.start_idx + idx
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == actual_idx:
                    entry = json.loads(line)
                    src, tgt = entry['english'], entry['chinese']
                    src_enc = self.tokenizer(src, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
                    tgt_enc = self.tokenizer(tgt, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
                    return {
                        'input_ids': src_enc['input_ids'].squeeze(),
                        'attention_mask': src_enc['attention_mask'].squeeze(),
                        'labels': tgt_enc['input_ids'].squeeze(),
                    }
        raise IndexError("Index out of range")

def load_dataset(data_path):
    # 加载与 BERT Base-uncased 对应的 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = TranslationDataset(data_path, tokenizer, is_train=True)
    val_dataset = TranslationDataset(data_path, tokenizer, is_train=False)
    
    return train_dataset, val_dataset
