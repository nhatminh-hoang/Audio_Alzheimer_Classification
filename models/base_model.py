import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        # Define your model layers here
        self.fc1 = nn.Linear(10, 5)  # Example: Input size 10, output size 5

    def forward(self, x):
        # Define the forward pass
        x = F.relu(self.fc1(x))  # Example: ReLU activation
        return x
    
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(drop_out)
        self.norm = nn.BatchNorm1d(hidden_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.norm(x)
        x = self.relu(x)        
        x = self.dropout(x)
        x = nn.Sigmoid()(self.fc2(x))
        return x

class CNNModel(nn.Module):
    def __init__(self, input_length=13, *args, **kwargs):
        """
        Args:
            input_length (int): Length of the input sequence.
        """
        super(CNNModel, self).__init__()
        
        # First Conv1d: input channels=1, output channels=64, kernel size=3
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)
        
        # Second Conv1d: input channels=64, output channels=128, kernel size=3
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)
        
        # To determine the flattened size after conv and pooling layers,
        # we use a dummy forward pass.
        dummy_input = torch.zeros(1, 1, input_length)
        out = F.relu(self.conv1(dummy_input))
        out = self.pool1(out)
        out = self.dropout1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = self.dropout2(out)
        self.flattened_size = out.numel()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        # Expect input x of shape (batch_size, length)
        # Reshape to (batch_size, channels=1, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
            
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        # Using sigmoid activation to produce output between 0 and 1
        x = torch.sigmoid(self.fc2(x))
        return x
    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(drop_out)
        self.norm = nn.LayerNorm(hidden_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = nn.Sigmoid()(self.fc(x[:, -1, :]))
        return x
    
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, 1)
        self.dropout = nn.Dropout(drop_out)
        self.norm = nn.LayerNorm(hidden_size*2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x[:, -1, :]))
        return x
    

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
    
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out, n_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.q = nn.Linear(input_size, hidden_size)
        self.k = nn.Linear(input_size, hidden_size)
        self.v = nn.Linear(input_size, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)
        self.dropout_p = drop_out
        self.dropout = nn.Dropout(drop_out)
        self.norm = nn.LayerNorm(output_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: batch_size, seq_len, hidden_size

        res = x.clone()
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Split the hidden_size into n_heads
        q = q.view(q.size(0), -1, self.n_heads, self.hidden_size//self.n_heads)
        k = k.view(k.size(0), -1, self.n_heads, self.hidden_size//self.n_heads)
        v = v.view(v.size(0), -1, self.n_heads, self.hidden_size//self.n_heads)

        # Transpose to get dimensions batch_size, n_heads, seq_len, hidden_size//n_heads
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate the attention
        # scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.hidden_size//self.n_heads)
        # scores = F.softmax(scores, dim=-1)
        # scores = self.dropout(scores)
        # output = torch.matmul(scores, v)
        output = scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)

        # Merge
        output = output.transpose(1, 2).contiguous().view(x.size(0), -1, self.hidden_size) # Shape batch_size, seq_len, hidden_size

        # Apply the output layer
        output = self.dropout(self.out(output))
        output = self.norm(output + res)
        return output
    
class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size*4)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(hidden_size*4, output_size)
        self.dropout = nn.Dropout(drop_out)
        self.norm = nn.LayerNorm(output_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x.clone()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.norm(x + res)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out, n_heads=4):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.mha = MultiHeadAttention(input_size, hidden_size, hidden_size, drop_out, n_heads)
        self.ffn = FFN(hidden_size, hidden_size*4, output_size, drop_out)

    def forward(self, x):
        x = self.mha(x)
        x = self.ffn(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out, seq_length=300, n_heads=4, n_layers=4):
        super(TransformerModel, self).__init__()
        self.emb = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(hidden_size, hidden_size, hidden_size, drop_out, n_heads) 
                                          for _ in range(n_layers)])
        self.fc = nn.Linear(hidden_size * seq_length, 1)

    def forward(self, x):

        x = self.emb(x)
        for l in self.layers:
            x = l(x)
        x = nn.Sigmoid()(self.fc(x.view(x.size(0), -1)))
        return x