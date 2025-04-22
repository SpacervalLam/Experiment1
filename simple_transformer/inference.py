"""
Transformer模型推理脚本

功能：
1. 加载训练好的模型
2. 处理输入数据
3. 生成预测
4. 输出结果
"""

import torch
from transformer import Transformer

class TransformerInference:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """初始化推理器
        
        参数:
            model_path: 模型文件路径
            device: 运行设备 (cuda/cpu)
        """
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # 使用与训练相同的配置
        self.config = {
            'src_vocab_size': 32000,
            'tgt_vocab_size': 32000,
            'd_model': 512,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 2048,
            'dropout': 0.1,
            'pad_idx': 0
        }
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        model = Transformer(
            src_vocab_size=self.config['src_vocab_size'],
            tgt_vocab_size=self.config['tgt_vocab_size'],
            d_model=self.config['d_model'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            d_ff=self.config['d_ff'],
            dropout=self.config['dropout']
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model.to(self.device)
    
    def preprocess(self, text):
        """预处理输入文本
        
        参数:
            text: 输入文本字符串
            
        返回:
            tokenized: 分词后的张量
        """
        # 这里应该实现实际的分词逻辑
        # 简化示例: 假设输入已经是token ID序列
        tokens = [int(t) for t in text.split() if t.isdigit()]
        return torch.tensor([tokens], dtype=torch.long, device=self.device)
    
    def postprocess(self, output_tensor):
        """后处理模型输出
        
        参数:
            output_tensor: 模型输出张量
            
        返回:
            text: 处理后的文本字符串
        """
        # 这里应该实现实际的detokenize逻辑
        # 简化示例: 直接返回token ID序列
        return ' '.join(str(t) for t in output_tensor.tolist()[0])
    
    def predict(self, src_text, max_length=50):
        """生成预测
        
        参数:
            src_text: 源文本
            max_length: 最大生成长度
            
        返回:
            预测结果文本
        """
        src = self.preprocess(src_text)
        
        # 初始化目标序列(以BOS token开始)
        tgt = torch.tensor([[1]], device=self.device)  # 假设1是BOS token
        
        with torch.no_grad():
            for _ in range(max_length):
                output = self.model(src, tgt)
                next_token = output.argmax(-1)[:, -1].unsqueeze(-1)
                tgt = torch.cat([tgt, next_token], dim=-1)
                
                # 遇到EOS token则停止
                if next_token.item() == 2:  # 假设2是EOS token
                    break
        
        return self.postprocess(tgt[:, 1:])  # 去掉BOS token

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Transformer模型推理')
    parser.add_argument('--model', type=str, default='checkpoints/final_model.pt',
                       help='模型文件路径')
    parser.add_argument('--text', type=str, required=True,
                       help='输入文本')
    parser.add_argument('--max_len', type=int, default=50,
                       help='最大生成长度')
    
    args = parser.parse_args()
    
    inferencer = TransformerInference(args.model)
    result = inferencer.predict(args.text, args.max_len)
    print("预测结果:", result)
