import torch
import torch.nn as nn
import numpy as np


# Informer
######################################################################################################
class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        # 分头计算
        batch_size = q.size(0)
        q = self.query(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return output

class Informer_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Informer_Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

class Informer_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Informer_Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)  # 假设输出大小与输入大小相同

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        
        prediction = self.fc(output)
        return prediction, hidden, cell


class Informer(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, pred_len, d_model, num_heads):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, num_heads)
        self.encoder = Informer_Encoder(input_dim, d_model)
        self.decoder = Informer_Decoder(d_model, d_model)
        self.fc = nn.Linear(d_model, input_dim)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def forward(self, x):
        output, hidden, cell = self.encoder(x)
        output = self.attention(output, output, output)
        output, _, _ = self.decoder(output, hidden, cell)
        decoded = self.fc(output[:, -self.pred_len:, :])
        return decoded


# DeepVAR
######################################################################################################

class VAR_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(VAR_Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell
    

class VAR_Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(VAR_Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell
    

class DeepVAR(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(DeepVAR, self).__init__()
        self.encoder = VAR_Encoder(input_dim, hidden_dim, num_layers)
        self.decoder = VAR_Decoder(input_dim, hidden_dim, output_dim, num_layers)
        
    def forward(self, src, trg = None, teacher_forcing_ratio=0.5):
        if trg == None:
            return self.predict(src,24)
        batch_size, trg_len, _ = trg.shape
        outputs = torch.zeros(batch_size, trg_len, trg.shape[2]).to(trg.device)
        
        hidden, cell = self.encoder(src)
        
        # 初始输入到解码器
        input = src[:, -1, :].unsqueeze(1)  # 取编码器的最后一个输出作为解码器的第一个输入
        
        for t in range(trg_len):
            # 插入输入token并接收输出
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            # 决定是否使用真实的下一个输入或使用预测的输出
            input = trg[:, t, :].unsqueeze(1) if torch.rand(1) < teacher_forcing_ratio else output
        
        return outputs
    

    def predict(self, src, future_steps):
        batch_size, _, input_dim = src.shape
        outputs = torch.zeros(batch_size, future_steps, input_dim).to(src.device)
        
        hidden, cell = self.encoder(src)
        
        input = src[:, -1, :].unsqueeze(1)
        
        for t in range(future_steps):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            input = output
        
        return outputs
    

# N-BEATS
######################################################################################################

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, hidden_units):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_units, input_size)  # Output is the same length as input

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

class NBeatsModel(nn.Module):
    def __init__(self, num_blocks, input_size, output_size, hidden_units):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.blocks = nn.ModuleList([NBeatsBlock(input_size, hidden_units) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(input_size * num_blocks, output_size)  # Adjust final output length

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input to (batch_size, input_size)
        forecasts = []
        for block in self.blocks:
            out = block(x)
            forecasts.append(out)
            x = x - out  # Residual connection
            
        forecast = torch.cat(forecasts, dim=-1)
        forecast = self.output_layer(forecast)  # Shape the output to desired output size
        forecast = forecast.view(forecast.size(0), 24, 8)  # Reshape to (batch_size, 24, 8)
        return forecast


# TFT
######################################################################################################

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.activation(self.fc1(x))
        transformed = self.fc2(hidden)
        gate = self.sigmoid(self.gate(hidden))

        if x.shape[-1] == transformed.shape[-1]:
            residual = x
        else:
            residual = nn.Linear(x.shape[-1], transformed.shape[-1], device=x.device)(x)

        output = gate * transformed + (1 - gate) * residual
        return output

class TemporalFusionTransformer(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, future_seq_length):
        super(TemporalFusionTransformer, self).__init__()
        self.encoder_lstm = nn.LSTM(feature_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.grn = GatedResidualNetwork(hidden_size, hidden_size, feature_size)
        self.future_seq_length = future_seq_length

    def forward(self, x):
        # Encoding
        encoded_outputs, (hidden, cell) = self.encoder_lstm(x)

        # Preparing the last hidden state for decoder input
        decoder_input = encoded_outputs[:, -self.future_seq_length:, :]

        # Decoding
        decoded_outputs, _ = self.decoder_lstm(decoder_input, (hidden, cell))

        # Applying GRN to each output step
        outputs = torch.stack([self.grn(decoded_outputs[:, i, :]) for i in range(decoded_outputs.shape[1])], dim=1)

        return outputs
