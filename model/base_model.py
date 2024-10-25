import torch
from torch import nn
import math
from .utils import GAT, GRUClassifier, CrossAttention, LSTMModel
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class ConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_sizes, dilation=2):
        super(ConvNet, self).__init__()
        layers = []
        for i in range(len(kernel_sizes)):
            dilation_size = dilation ** i
            kernel_size = kernel_sizes[i]
            padding = (kernel_size - 1) * dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding),
                nn.BatchNorm1d(out_channels), nn.ReLU(), Chomp1d(padding)]

        self.network = nn.Sequential(*layers)

        self.out_dim = num_channels[-1]

    def forward(self, x):
        x = x.permute(0, 2, 1).float()
        out = self.network(x)
        out = out.permute(0, 2, 1)
        return out


class SelfAttention(nn.Module):
    def __init__(self, input_size, seq_len):
        super(SelfAttention, self).__init__()
        self.atten_w = nn.Parameter(torch.randn(seq_len, input_size, 1))
        self.atten_bias = nn.Parameter(torch.randn(seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.atten_bias.data.fill_(0)

    def forward(self, x):
        input_tensor = x.transpose(1, 0)  # w x b x h
        input_tensor = (torch.bmm(input_tensor, self.atten_w) + self.atten_bias)  # w x b x out
        input_tensor = input_tensor.transpose(1, 0)
        atten_weight = input_tensor.tanh()
        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), x).squeeze()
        return weighted_sum

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(7.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)


class GaussianAttention(nn.Module):
    def __init__(self, input_size, seq_len):
        super(GaussianAttention, self).__init__()
        self.atten_w = nn.Parameter(torch.randn(seq_len, input_size, 1))
        self.atten_bias = nn.Parameter(torch.randn(seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.atten_bias.data.fill_(0)
        self.seq_len = seq_len
        self.batch_norm = nn.BatchNorm1d(input_size)

        # Gaussian parameters
        self.mu = torch.tensor(seq_len // 2, dtype=torch.float)  # initialized at the center
        self.sigma = torch.tensor(5.0, dtype=torch.float)  # initialized with 5

    def forward(self, x):
        # x: [batch_size, window_size, input_size]
        batch_size = x.size(0)
        input_size = x.size(2)

        # Calculate Gaussian weights
        positions = torch.arange(self.seq_len, dtype=torch.float).unsqueeze(1).to(x.device)  # Shape: (w, 1)
        gauss_weight = 1.0 / (math.sqrt(2 * math.pi) * self.sigma) * torch.exp(
            -0.5 * ((positions - self.mu) / self.sigma) ** 2).repeat(1, input_size)
        gauss_weight = gauss_weight.unsqueeze(0).repeat(batch_size, 1, 1)  # Normalize w*input

        x = x.transpose(1, 2)  # x: [batch_size, input_size, window_size]
        x = self.batch_norm(x)
        x = x.transpose(1, 2)  # x: [batch_size, window_size, input_size]
        x = x + gauss_weight

        input_tensor = x.transpose(1, 0)  # w x b x h
        input_tensor = (torch.bmm(input_tensor, self.atten_w) + self.atten_bias)  # w x (b x out)
        input_tensor = input_tensor.transpose(1, 0)  # w x b x out
        atten_weight = input_tensor.tanh()  #
        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), x).squeeze()
        return weighted_sum

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)


class MetricModel(nn.Module):
    def __init__(self, metric_num, metric_hiddens=[64], metric_kernel_sizes=[2], self_attn=True,
                 chunk_lenth=41):
        super(MetricModel, self).__init__()
        self.metric_num = metric_num
        self.out_dim = metric_hiddens[-1]
        in_dim = metric_num

        assert len(metric_hiddens) == len(metric_kernel_sizes)
        self.net = ConvNet(num_inputs=in_dim, num_channels=metric_hiddens, kernel_sizes=metric_kernel_sizes)

        self.self_attn = self_attn
        if self_attn:
            assert (chunk_lenth is not None)
            self.attn_layer = GaussianAttention(self.out_dim, chunk_lenth)

    def forward(self, x):
        assert x.shape[-1] == self.metric_num
        hidden_states = self.net(x)
        if self.self_attn:
            return self.attn_layer(hidden_states)
        return hidden_states[:, -1, :]


class LogModel(nn.Module):
    def __init__(self, embedding_size, out_dim=64):
        super(LogModel, self).__init__()
        self.lstm = LSTMModel(embedding_size, out_dim)

    def forward(self, x, perm_idx):  # [bz, event_num]
        """
        Input:
            paras: mu with length of event_num
        """
        x = self.lstm(x)
        _, unperm_idx = perm_idx.sort(0)
        x = x[unperm_idx]
        return x


class TraceModel(nn.Module):
    def __init__(self, embedding_size, out_dim=64):
        super(TraceModel, self).__init__()
        self.encoder1 = nn.Linear(1, out_dim)
        self.encoder2 = nn.Linear(1, out_dim)
        self.encoder3 = GAT(embedding_size, out_dim)

        self.fc1 = nn.Linear(3 * out_dim, 3 * 128)
        self.fc2 = nn.Linear(3 * 128, 128)
        self.fc3 = nn.Linear(128, out_dim)

    def forward(self, x, edge_index):  # [bz, event_num]
        """
        Input:
            paras: mu with length of event_num
        """
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x3 = x[:, 2:]
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x3 = self.encoder3(x3, edge_index)
        x = torch.cat((x1, x2, x3), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class base_model(nn.Module):
    def __init__(self, embedding_size, metric_feature_size, num_classes, gat_output_size=64):
        super(base_model, self).__init__()
        self.trace_feature_extractor = TraceModel(embedding_size, 64)
        self.log_feature_extractor = LogModel(embedding_size, out_dim=64)
        self.metric_feature_extractor = MetricModel(metric_feature_size, metric_hiddens=[64])

        self.trace_attention = CrossAttention(64)
        self.log_attention = CrossAttention(64)
        self.metric_attention = CrossAttention(64)

        self.fuse = nn.Linear(6 * 64, 128)
        self.activate = nn.GLU()

        self.classifier = GRUClassifier(gat_output_size, num_classes)

    def forward(self, trace_feature, log_feature, metric_feature, perm_idx, edge_index, node_length):
        trace_x = self.trace_feature_extractor(trace_feature, edge_index)
        log_x = self.log_feature_extractor(log_feature, perm_idx)
        metric_x = self.metric_feature_extractor(metric_feature[:, :, :37])

        x1 = self.trace_attention(trace_x, log_x)
        x2 = self.trace_attention(trace_x, metric_x)
        x3 = self.log_attention(log_x, trace_x)
        x4 = self.log_attention(log_x, metric_x)
        x5 = self.metric_attention(metric_x, trace_x)
        x6 = self.metric_attention(metric_x, log_x)

        fusion_feature = self.activate(self.fuse(torch.cat((x1, x2, x3, x4, x5, x6), dim=-1)))

        split_node_tensor = torch.split(fusion_feature, node_length)
        # padding
        sequence_lengths = [len(seq) for seq in split_node_tensor]
        sequence_lengths, perm_idx = torch.sort(torch.LongTensor(sequence_lengths), descending=True)
        split_node_tensor = [split_node_tensor[i] for i in perm_idx]
        max_length = max(sequence_lengths)
        padded_sequences = torch.zeros(len(split_node_tensor), max_length, split_node_tensor[0].shape[1])

        for i, seq in enumerate(split_node_tensor):
            end = sequence_lengths[i]
            padded_sequences[i, :end, :] = seq
        log_packed_input = pack_padded_sequence(padded_sequences, sequence_lengths, batch_first=True).cuda()
        output = self.classifier(log_packed_input)
        _, unperm_idx = perm_idx.sort(0)
        output = output[unperm_idx]

        return output
