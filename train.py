import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torchaudio.transforms as transforms
from tqdm import tqdm


class AudioFolderDataset(Dataset):
    def __init__(self, root_dir, target_sample_rate=8000, target_length=56000):
        self.root_dir = root_dir
        self.file_paths = []
        self.labels = []
        self.label_map = {}
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length

        self.mel_spectrogram = transforms.MelSpectrogram(sample_rate=target_sample_rate, n_mels=64)
        self._load_dataset()

    def _load_dataset(self):
        label_names = os.listdir(self.root_dir)
        for idx, label_name in enumerate(label_names):
            label_dir = os.path.join(self.root_dir, label_name)
            if os.path.isdir(label_dir):
                self.label_map[label_name] = idx
                for file_name in os.listdir(label_dir):
                    if file_name.endswith('.wav'):
                        self.file_paths.append(os.path.join(label_dir, file_name))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, sample_rate = torchaudio.load(file_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = self.preprocess(waveform, sample_rate)

        mel_spec = self.mel_spectrogram(waveform)
        return mel_spec, label

    def preprocess(self, waveform, sample_rate):
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)(waveform)

        waveform = self.pad_or_trim_waveform(waveform)

        return waveform

    def pad_or_trim_waveform(self, waveform):
        current_length = waveform.shape[1]
        if current_length > self.target_length:
            waveform = waveform[:, :self.target_length]
        elif current_length < self.target_length:
            padding = self.target_length - current_length
            waveform = F.pad(waveform, (0, padding))

        return waveform


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
            attention_scores = torch.matmul(attention_scores,
                                            lstm_out.transpose(1, 2))
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


def main():
    data_dir = './dataset/'
    dataset = AudioFolderDataset(root_dir=data_dir, target_sample_rate=8000,
                                 target_length=56000)

    seed = 42
    torch.manual_seed(seed)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    num_classes = 9
    model = CRNNAttentionClassifier(num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 150
    best_val_accuracy = 0.0
    best_model_path = './model/model_attention.pth'

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for inputs, labels in train_loader_tqdm:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

        model.eval()
        all_preds = []
        all_labels = []

        val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for inputs, labels in val_loader_tqdm:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(all_labels, all_preds)
        print(f'Validation Accuracy: {val_accuracy}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved Best Model with Accuracy: {val_accuracy}')

    best_model_path = './model/model_attention.pth'

    num_classes = 9
    model = CRNNAttentionClassifier(num_classes=num_classes)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy}')


if __name__ == "__main__":
    main()
