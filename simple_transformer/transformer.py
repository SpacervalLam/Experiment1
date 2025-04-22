"""
基于论文"Attention Is All You Need"(https://arxiv.org/abs/1706.03762)的Transformer模型复现

实现步骤：
1. 多头注意力机制(MultiHeadAttention)
2. 位置前馈网络(PositionWiseFeedForward) 
3. 位置编码(PositionalEncoding)
4. 编码器层(EncoderLayer)
5. 解码器层(DecoderLayer)
6. 完整Transformer模型
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """多头注意力机制实现
    
    参数:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        
    属性:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        head_dim (int): 每个头的维度
        wq (nn.Linear): Query线性变换层
        wk (nn.Linear): Key线性变换层  
        wv (nn.Linear): Value线性变换层
        wo (nn.Linear): 输出线性变换层
    """
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "模型维度必须能被头数整除"
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        参数:
            q (torch.Tensor): Query张量，形状(batch_size, seq_len, d_model)
            k (torch.Tensor): Key张量，形状(batch_size, seq_len, d_model)
            v (torch.Tensor): Value张量，形状(batch_size, seq_len, d_model)
            mask (Optional[torch.Tensor]): 注意力掩码
            
        返回:
            torch.Tensor: 注意力输出，形状(batch_size, seq_len, d_model)
        """
        batch_size = q.size(0) # 批量大小
        
        # Linear projections
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        # Split into multiple heads
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # 计算注意力分数矩阵（缩放）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # 掩盖pad部分的注意力分数
        attention = F.softmax(scores, dim=-1) # Softmax归一化获得注意力权重
        output = torch.matmul(attention, v) # 将注意力权重应用到V上
        
        # Concatenate heads back to original shape
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.wo(output)

class PositionWiseFeedForward(nn.Module):
    """位置前馈网络实现
    
    参数:
        d_model (int): 输入和输出的维度
        d_ff (int): 内部隐藏层的维度
        
    属性:
        fc1 (nn.Linear): 第一个线性层，将维度从d_model扩展到d_ff
        fc2 (nn.Linear): 第二个线性层，将维度从d_ff压缩回d_model
        dropout (nn.Dropout): Dropout层
    """
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状(batch_size, seq_len, d_model)
            
        返回:
            torch.Tensor: 输出张量，形状(batch_size, seq_len, d_model)
        """
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class PositionalEncoding(nn.Module):
    """位置编码实现
    
    参数:
        d_model (int): 编码维度
        max_len (int): 最大序列长度
        
    属性:
        pe (torch.Tensor): 位置编码矩阵，形状(1, max_len, d_model)
    """
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状(batch_size, seq_len, d_model)
            
        返回:
            torch.Tensor: 添加位置编码后的张量，形状(batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    """Transformer编码器层实现
    
    参数:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        d_ff (int): 前馈网络内部维度
        dropout (float): Dropout概率
        
    属性:
        self_attn (MultiHeadAttention): 多头自注意力层
        feed_forward (PositionWiseFeedForward): 前馈网络层
        norm1 (nn.LayerNorm): 第一层归一化
        norm2 (nn.LayerNorm): 第二层归一化
        dropout (nn.Dropout): Dropout层
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状(batch_size, seq_len, d_model)
            mask (Optional[torch.Tensor]): 注意力掩码
            
        返回:
            torch.Tensor: 输出张量，形状(batch_size, seq_len, d_model)
        """
        # 第一子层: 多头自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 第二子层: 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    """Transformer解码器层实现
    
    参数:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        d_ff (int): 前馈网络内部维度
        dropout (float): Dropout概率
        
    属性:
        self_attn (MultiHeadAttention): 多头自注意力层
        cross_attn (MultiHeadAttention): 多头交叉注意力层
        feed_forward (PositionWiseFeedForward): 前馈网络层
        norm1 (nn.LayerNorm): 第一层归一化
        norm2 (nn.LayerNorm): 第二层归一化
        norm3 (nn.LayerNorm): 第三层归一化
        dropout (nn.Dropout): Dropout层
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状(batch_size, seq_len, d_model)
            encoder_output (torch.Tensor): 编码器输出，形状(batch_size, seq_len, d_model)
            src_mask (Optional[torch.Tensor]): 源序列注意力掩码
            tgt_mask (Optional[torch.Tensor]): 目标序列注意力掩码
            
        返回:
            torch.Tensor: 输出张量，形状(batch_size, seq_len, d_model)
        """
        # 第一子层: 多头自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 第二子层: 多头交叉注意力 + 残差连接 + 层归一化
        attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 第三子层: 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    """完整的Transformer模型实现
    
    参数:
        src_vocab_size (int): 源语言词汇表大小
        tgt_vocab_size (int): 目标语言词汇表大小
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        num_layers (int): 编码器/解码器层数
        d_ff (int): 前馈网络内部维度
        dropout (float): Dropout概率
        
    属性:
        encoder_embed (nn.Embedding): 编码器输入嵌入层
        decoder_embed (nn.Embedding): 解码器输入嵌入层
        pos_encoding (PositionalEncoding): 位置编码层
        encoder_layers (nn.ModuleList): 编码器层列表
        decoder_layers (nn.ModuleList): 解码器层列表
        fc_out (nn.Linear): 最终线性输出层
        dropout (nn.Dropout): Dropout层
    """
    @staticmethod
    def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
        """创建padding mask
        
        参数:
            seq (torch.Tensor): 输入序列，形状(batch_size, seq_len)
            pad_idx (int): padding token的索引
            
        返回:
            torch.Tensor: 注意力掩码，形状(batch_size, 1, 1, seq_len)
        """
        mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    @staticmethod
    def create_causal_mask(seq_len: int) -> torch.Tensor:
        """创建causal mask
        
        参数:
            seq_len (int): 序列长度
            
        返回:
            torch.Tensor: 注意力掩码，形状(seq_len, seq_len)
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_pad_idx: Optional[int] = None,
        tgt_pad_idx: Optional[int] = None
    ) -> torch.Tensor:
        """前向传播
        
        参数:
            src (torch.Tensor): 源序列输入，形状(batch_size, src_seq_len)
            tgt (torch.Tensor): 目标序列输入，形状(batch_size, tgt_seq_len)
            src_pad_idx (Optional[int]): 源序列padding token索引
            tgt_pad_idx (Optional[int]): 目标序列padding token索引
            
        返回:
            torch.Tensor: 模型输出，形状(batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # 生成mask
        src_mask = None
        tgt_mask = None
        if src_pad_idx is not None:
            src_mask = self.create_padding_mask(src, src_pad_idx)
        if tgt_pad_idx is not None:
            tgt_mask = self.create_padding_mask(tgt, tgt_pad_idx)
            # 解码器自注意力需要causal mask
            seq_len = tgt.size(1)
            causal_mask = self.create_causal_mask(seq_len).to(tgt.device)
            tgt_mask = tgt_mask | causal_mask.unsqueeze(0).unsqueeze(0)
        
        src_embed = self.dropout(self.pos_encoding(self.encoder_embed(src)))
        tgt_embed = self.dropout(self.pos_encoding(self.decoder_embed(tgt)))
        
        # 编码器处理
        src_output = src_embed
        for layer in self.encoder_layers:
            src_output = layer(src_output, src_mask)
        
        # 解码器处理
        tgt_output = tgt_embed
        for layer in self.decoder_layers:
            tgt_output = layer(tgt_output, src_output, src_mask, tgt_mask)
        
        output = self.fc_out(tgt_output)
        return output
