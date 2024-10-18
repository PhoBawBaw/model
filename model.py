import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SeqSelfAttention(nn.Module):
    def __init__(self, attention_size=128, attention_type='multiplicative', dropout=0.3):
        super(SeqSelfAttention, self).__init__()
        self.attention_type = attention_type
        self.attention_size = attention_size
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

        if attention_type == 'multiplicative':
            self.attention_weights = nn.Parameter(torch.FloatTensor(attention_size, attention_size))
        elif attention_type == 'additive':
            self.W_q = nn.Linear(attention_size, attention_size)
            self.W_k = nn.Linear(attention_size, attention_size)
            self.V = nn.Linear(attention_size, 1)

        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, lstm_out):
        batch_size, seq_length, hidden_size = lstm_out.size()

        if self.attention_type == 'multiplicative':
            attention_scores = torch.matmul(lstm_out, self.attention_weights)
            attention_scores = torch.matmul(attention_scores, lstm_out.transpose(1, 2))
            attention_scores = self.tanh(attention_scores)
            attention_weights = F.softmax(attention_scores, dim=-1)

        else:
            query = self.W_q(lstm_out)
            key = self.W_k(lstm_out)
            scores = self.tanh(self.V(query + key))
            attention_weights = F.softmax(scores, dim=1)

        attention_output = torch.matmul(attention_weights, lstm_out)
        return attention_output


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        batch_size, time_steps, *input_shape = x.size()
        x = x.contiguous().view(-1, *input_shape)
        x = self.module(x)
        x = x.view(batch_size, time_steps, -1)
        return x


class CRNNAttentionClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size=128, attention_size=128, lstm_layers=1):
        super(CRNNAttentionClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(3, 3)
        self.dropout1 = nn.Dropout(0.3)

        self.cnn_to_lstm = nn.Linear(2688, 128)

        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True,
                            bidirectional=False)

        self.attention = SeqSelfAttention(attention_size=attention_size, attention_type='multiplicative', dropout=0.3)

        self.time_distributed_dense1 = TimeDistributed(nn.Linear(hidden_size, 64))
        self.time_distributed_dropout = TimeDistributed(nn.Dropout(0.3))
        self.time_distributed_flatten = TimeDistributed(nn.Flatten())

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout1(x)

        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.view(batch_size, width, -1)

        x = self.cnn_to_lstm(x)

        lstm_out, _ = self.lstm(x)
        attention_out = self.attention(lstm_out)

        x = self.time_distributed_dense1(attention_out)
        x = self.time_distributed_dropout(x)
        x = self.time_distributed_flatten(x)

        out = self.fc(x.mean(dim=1))
        return out
