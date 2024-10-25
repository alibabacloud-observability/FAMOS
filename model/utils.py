import torch.nn as nn
from torch_geometric.nn import GATConv


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        # Define linear transformations for query, key, and value
        self.query_transform = nn.Linear(feature_dim, feature_dim)
        self.key_transform = nn.Linear(feature_dim, feature_dim)
        self.value_transform = nn.Linear(feature_dim, feature_dim)
        # Multi-head attention
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key_value):
        # Query, key, and value pass through the corresponding linear layers
        q = self.query_transform(query)
        k = self.key_transform(key_value)
        v = self.value_transform(key_value)
        # Apply the multi-head attention mechanism
        # MultiheadAttention expects input shapes to be [batch_size, seq_len, feature_dim]. Since we have only one feature vector, seq_len=1
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        attention_output, _ = self.multi_head_attention(q, k, v)  # [batch_size, seq_len, feature_dim]
        attention_output = attention_output.squeeze(1)
        return attention_output


class GAT(nn.Module):
    def __init__(self, input_size, output_size, num_heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_size, output_size, heads=num_heads, concat=False)

    def forward(self, x, edge_index):
        out = self.conv1(x, edge_index)
        return out


class GRUClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        final_hidden_state = h_n[-1]
        output = self.fc(final_hidden_state)
        return output


class LSTMModel(nn.Module):
    def __init__(self, input_size, align_size, num_layers=2, hidden_size=128):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Linear layer maps the last hidden state of the LSTM to the embedding feature size
        self.linear = nn.Linear(hidden_size, align_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        final_hidden_state = h_n[-1]
        embedding = self.linear(final_hidden_state)
        return embedding
