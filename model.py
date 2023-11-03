import torch
from torch import nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, in_size, out_size):
        """Construct LSTM model.
        """
        super().__init__()
        self.outSize = out_size
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True,
                            dropout=0.2)
        self.bn1 = nn.BatchNorm1d(self.outSize)
        self.linear1 = nn.Linear(in_features=512, out_features=1024)
        self.dropout1 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm1d(self.outSize)
        self.linear2 = nn.Linear(in_features=1024, out_features=256)
        self.dropout2 = nn.Dropout(0.2)
        self.bn3 = nn.BatchNorm1d(self.outSize)
        self.linear3 = nn.Linear(in_features=256, out_features=161)

        self.activation = nn.Sigmoid()

    def forward(self, model_input):
        x, h = self.lstm(model_input)
        x = self.bn1(x)
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)

        # continue with old
        x = self.bn2(x)

        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.bn3(x)
        x = self.linear3(x)

        x = self.activation(x)
        # Reshape to be real and imaginary components
        x = x.view(-1, self.outSize, 161)

        return x
    



class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / self.head_dim**0.5
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = torch.nn.functional.softmax(attention, dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class LSTMModelAttention(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.outSize = out_size
        
        # Since you don't want to reduce the freq axis, ensure padding is added to keep the dimensions.
        self.conv1 = nn.Conv1d(161, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        
        # LSTM will keep the sequence length (822) unchanged.
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
        self.self_attention = SelfAttention(embed_size=512, heads=8)
        
        self.bn1 = nn.BatchNorm1d(822)  # Adjust batch norm size
        self.linear1 = nn.Linear(512, 1024)
        self.dropout1 = nn.Dropout(0.2)
        
        self.bn2 = nn.BatchNorm1d(822)  # Adjust batch norm size
        self.linear2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(0.2)
        
        self.bn3 = nn.BatchNorm1d(822)  # Adjust batch norm size
        self.linear3 = nn.Linear(256, 161)  # Set the output size to 161 to match frequency axis

        self.activation = nn.Sigmoid()

    def forward(self, model_input):
        # Transpose model_input to have channels as the second dimension
        model_input = model_input.transpose(1, 2)
        
        x = F.relu(self.conv1(model_input))
        x = F.relu(self.conv2(x))
        
        # Transpose again to have sequence_length as the second dimension before LSTM
        x = x.transpose(1, 2)
        
        x, _ = self.lstm(x)
        
        x = self.self_attention(x, x, x, mask=None)
        
        x = self.bn1(x)
        x = self.linear1(x)
        x = x.view(x.size(0), x.size(1), -1)  # Ensure 3D shape for following operations
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.bn2(x)
        x = self.linear2(x)
        x = x.view(x.size(0), x.size(1), -1)  # Ensure 3D shape for following operations
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.bn3(x)
        x = self.linear3(x)
        x = self.activation(x)

        # No need for reshaping since the desired output shape is maintained throughout
        return x


class EnhancedLSTMModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(EnhancedLSTMModel, self).__init__()

        self.outSize = out_size

        self.conv1 = nn.Conv1d(161, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)

        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)

        self.self_attention = SelfAttention(embed_size=512, heads=8)

        self.bn1 = nn.BatchNorm1d(822)
        self.linear1 = nn.Linear(512, 1024)
        self.dropout1 = nn.Dropout(0.3)

        self.bn2 = nn.BatchNorm1d(822)
        self.linear2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(0.3)

        self.bn3 = nn.BatchNorm1d(822)
        self.linear3 = nn.Linear(256, 161)

        self.activation = nn.Sigmoid()

    def forward(self, model_input):
        model_input = model_input.transpose(1, 2)

        x = F.relu(self.conv1(model_input))
        x = F.relu(self.conv2(x))

        x = x.transpose(1, 2)
        x, _ = self.lstm(x)

        x = self.self_attention(x, x, x, mask=None)
        x = self.bn1(x)
        x = self.linear1(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.bn2(x)
        x = self.linear2(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.bn3(x)
        x = self.linear3(x)
        x = self.activation(x)

        return x
