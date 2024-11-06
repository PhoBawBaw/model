import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torchaudio.transforms as transforms
from tqdm import tqdm
import random
import bentoml
import psycopg2
import shutil


# 데이터셋 클래스 정의
class AudioFolderDataset(Dataset):
    def __init__(self, root_dir, target_sample_rate=8000, target_length=56000):
        self.root_dir = root_dir
        self.file_paths = []
        self.labels = []
        self.label_map = {}
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length

        # Mel Spectrogram 변환기
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

        # 오디오 파일 로드
        waveform, sample_rate = torchaudio.load(file_path)

        # 항상 모노로 변환 (채널이 여러 개일 경우 평균)
        if waveform.shape[0] > 1:  # 스테레오(2 채널)일 경우
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 전처리 함수 적용
        waveform = self.preprocess(waveform, sample_rate)

        # Mel Spectrogram으로 변환
        mel_spec = self.mel_spectrogram(waveform)
        return mel_spec, label

    def preprocess(self, waveform, sample_rate):
        # 리샘플링
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)(waveform)

        # 길이 맞추기
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


# Attention Layer 정의
class SeqSelfAttention(nn.Module):
    def __init__(self, attention_size=128, attention_type='multiplicative', dropout=0.3):
        super(SeqSelfAttention, self).__init__()
        self.attention_type = attention_type
        self.attention_size = attention_size
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

        if attention_type == 'multiplicative':
            # attention_weights 크기: (attention_size, attention_size), 즉 (128, 128)
            self.attention_weights = nn.Parameter(torch.FloatTensor(attention_size, attention_size))
        elif attention_type == 'additive':
            self.W_q = nn.Linear(attention_size, attention_size)
            self.W_k = nn.Linear(attention_size, attention_size)
            self.V = nn.Linear(attention_size, 1)

        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, lstm_out):
        # LSTM 출력 크기: [batch_size, seq_length, hidden_size]
        batch_size, seq_length, hidden_size = lstm_out.size()

        if self.attention_type == 'multiplicative':
            # attention_weights 크기: (hidden_size, hidden_size) = (128, 128)
            attention_scores = torch.matmul(lstm_out, self.attention_weights)  # [batch_size, seq_length, hidden_size]
            attention_scores = torch.matmul(attention_scores,
                                            lstm_out.transpose(1, 2))  # [batch_size, seq_length, seq_length]
            attention_scores = self.tanh(attention_scores)
            attention_weights = F.softmax(attention_scores, dim=-1)

        else:  # Additive Attention (추가 옵션)
            query = self.W_q(lstm_out)
            key = self.W_k(lstm_out)
            scores = self.tanh(self.V(query + key))
            attention_weights = F.softmax(scores, dim=1)

        attention_output = torch.matmul(attention_weights, lstm_out)  # [batch_size, seq_length, hidden_size]
        return attention_output


# TimeDistributed 레이어 정의 (각 타임 스텝별로 동일한 레이어 적용)
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        batch_size, time_steps, *input_shape = x.size()
        x = x.contiguous().view(-1, *input_shape)  # (batch * time_steps, input_shape)
        x = self.module(x)
        x = x.view(batch_size, time_steps, -1)  # (batch, time_steps, output_shape)
        return x


# CRNN + Attention 모델 정의
class CRNNAttentionClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size=128, attention_size=128, lstm_layers=1):
        super(CRNNAttentionClassifier, self).__init__()

        # CNN 레이어
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(3, 3)
        self.dropout1 = nn.Dropout(0.3)

        # CNN 출력 차원을 줄이기 위한 Linear 레이어
        self.cnn_to_lstm = nn.Linear(2688, 128)

        # LSTM 레이어
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True,
                            bidirectional=False)

        # Attention 레이어
        self.attention = SeqSelfAttention(attention_size=attention_size, attention_type='multiplicative', dropout=0.3)

        # TimeDistributed 레이어
        self.time_distributed_dense1 = TimeDistributed(nn.Linear(hidden_size, 64))
        self.time_distributed_dropout = TimeDistributed(nn.Dropout(0.3))
        self.time_distributed_flatten = TimeDistributed(nn.Flatten())

        # 최종 출력 레이어
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # CNN 부분
        x = F.relu(self.bn1(self.conv1(x)))  # Conv -> BatchNorm -> ReLU
        x = self.pool(x)  # Max Pooling
        x = self.dropout1(x)  # Dropout

        # CNN 출력 텐서의 형태를 변경하여 LSTM에 전달 (batch, time_steps, features)
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)  # [batch_size, width, channels, height]
        x = x.view(batch_size, width, -1)  # [batch_size, time_steps, features]

        # Linear 레이어로 차원을 LSTM input_size에 맞춤 (128)
        x = self.cnn_to_lstm(x)

        # LSTM 부분
        lstm_out, _ = self.lstm(x)  # LSTM 적용
        attention_out = self.attention(lstm_out)  # Attention 적용

        # TimeDistributed 레이어 적용
        x = self.time_distributed_dense1(attention_out)
        x = self.time_distributed_dropout(x)
        x = self.time_distributed_flatten(x)

        # 최종 출력
        out = self.fc(x.mean(dim=1))  # 각 타임 스텝의 평균을 취하여 최종 출력
        return out


def stateful_training(data_chunks, test_loader, num_classes):
    K = len(data_chunks)
    # BentoML 모델 저장소에서 최신 모델 로드 시도
    try:
        latest_model = bentoml.pytorch.get("crnn_attention_classifier:latest").to_runner().model
        model = CRNNAttentionClassifier(num_classes=num_classes)
        model.load_state_dict(latest_model.state_dict())
        print("Loaded latest model from BentoML model store.")
    except bentoml.exceptions.NotFound:
        # 모델이 없으면 새로 초기화
        model = CRNNAttentionClassifier(num_classes=num_classes)
        print("No existing model found. Initialized a new model.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for t in range(K):
        print(f"\n[Stateful Training] Time Step {t + 1}/{K}")
        training_data = data_chunks[t]
        train_loader = DataLoader(training_data, batch_size=16, shuffle=True)

        num_epochs = 5
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
                train_loader_tqdm.set_postfix(loss=running_loss / len(train_loader))
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

        # 모델 저장
        bentoml.pytorch.save_model(
            name="crnn_attention_classifier",
            model=model,
            signatures={"__call__": {"batchable": True}},
            metadata={"time_step": t + 1}
        )
        print(f"Saved model at time step {t + 1} to BentoML model store.")

        # 테스트 세트로 평가
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
        print(f"Test Accuracy at Time Step {t + 1}: {accuracy:.4f}")


def fetch_audio_filenames_and_labels():
    # PostgreSQL 연결 설정
    conn = psycopg2.connect(
        dbname='',
        user='',
        password='',
        host='',
        port=''
    )
    cur = conn.cursor()
    # audio와 label 컬럼의 파일명 가져오기
    cur.execute("SELECT audio, label FROM continual")  # Assuming 'label' is the name of the label column
    audio_files_with_labels = cur.fetchall()
    cur.close()
    conn.close()
    return audio_files_with_labels  # 결과를 리스트로 변환


def copy_audio_files_with_labels(file_list, source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for file_name, label in file_list:
        # Label 디렉토리 생성
        label_dir = os.path.join(target_dir, label)  # Using the label as directory name
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(label_dir, file_name)
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
        else:
            print(f"File {source_path} does not exist.")


def main():
    data_dir = '/baby/audio'
    target_dir = '/app/annotated'

    audio_files_with_labels = fetch_audio_filenames_and_labels()  # Fetching audio files and labels
    copy_audio_files_with_labels(audio_files_with_labels, data_dir, target_dir)  # Copying files with labels

    # Set up the dataset
    dataset = AudioFolderDataset(root_dir=target_dir, target_sample_rate=8000, target_length=56000)

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)

    total_size = len(dataset)
    test_size = int(0.2 * total_size)
    train_size = total_size - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    K = 5
    chunk_size = train_size // K
    chunk_sizes = [chunk_size] * K
    chunk_sizes[-1] += train_size % K
    data_chunks = random_split(train_dataset, chunk_sizes)

    num_classes = len(dataset.label_map)

    # Stateful Training 실행
    print("\n=== Stateful Training ===")
    stateful_training(data_chunks, test_loader, num_classes)


if __name__ == "__main__":
    main()
