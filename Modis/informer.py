import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = torch.matmul(weights, value)
        return output, weights

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout_rate):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout_rate)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attention_output, _ = self.attention(query, key, value, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.out_proj(attention_output)
        return output
    

class ProbSparseSelfAttention(MultiHeadAttention):
    def __init__(self, num_heads, d_model, dropout_rate, attention_sampling_rate=0.1):
        super().__init__(num_heads, d_model, dropout_rate)
        self.attention_sampling_rate = attention_sampling_rate

    def forward(self, query, key, value, mask=None):
        batch_size, seq_length, d_model = key.size()
        num_samples = int(seq_length * self.attention_sampling_rate)

        # Sampling strategy: random sampling
        # Here we use torch.randint for simplicity; in practice, you might want to sample based on some criteria
        sampled_key_indices = torch.randint(seq_length, (batch_size, num_samples), device=query.device)
        key_sampled = key.gather(1, sampled_key_indices.unsqueeze(-1).expand(-1, -1, d_model))
        value_sampled = value.gather(1, sampled_key_indices.unsqueeze(-1).expand(-1, -1, d_model))

        # Proceed with the usual MultiHeadAttention using the sampled keys and values
        return super().forward(query, key_sampled, value_sampled, mask)
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout_rate=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x
    

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, attention_sampling_rate):
        super().__init__()
        self.attention = ProbSparseSelfAttention(num_heads, d_model, dropout_rate, attention_sampling_rate)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src, src_mask=None):
        src2 = self.norm1(src)
        src = src + self.dropout(self.attention(src2, src2, src2, src_mask))
        src2 = self.norm2(src)
        src = src + self.dropout(self.feed_forward(src2))
        return src
    
class InformerEncoder(nn.Module):
    def __init__(self, layer, N, d_model):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return self.norm(src)
    
class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, attention_sampling_rate):
        super(InformerDecoderLayer, self).__init__()
        self.self_attention = ProbSparseSelfAttention(num_heads, d_model, dropout_rate, attention_sampling_rate)
        self.cross_attention = MultiHeadAttention(num_heads, d_model, dropout_rate)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2 = self.norm1(tgt)
        tgt = tgt + self.dropout(self.self_attention(tgt2, tgt2, tgt2, tgt_mask))
        tgt2 = self.norm2(tgt)
        tgt = tgt + self.dropout(self.cross_attention(tgt2, memory, memory, memory_mask))
        tgt2 = self.norm3(tgt)
        tgt = tgt + self.dropout(self.feed_forward(tgt2))
        return tgt
    

class InformerDecoder(nn.Module):
    def __init__(self, layer, N, d_model):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.norm(tgt)
    

class Informer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Informer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  # 输入序列嵌入
        self.tgt_embed = tgt_embed  # 目标序列嵌入
        self.generator = generator  # 输出层

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(self.src_embed(src), src_mask)
        output = self.decoder(self.tgt_embed(tgt), memory, tgt_mask, memory_mask)
        return self.generator(output)

class Generator(nn.Module):
    def __init__(self, d_model, output_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, output_size)

    def forward(self, x):
        return self.proj(x)