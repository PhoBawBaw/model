import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


# 데이터셋 클래스 정의
class AudioFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_sample_rate=8000, target_length=8000):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []
        self.label_map = {}
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length

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

        # 항상 모노로 변환
        if waveform.shape[0] > 1:  # 스테레오(2 채널)일 경우
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 전처리 함수 적용
        waveform = self.preprocess(waveform, sample_rate)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

    def preprocess(self, waveform, sample_rate):
        # 리샘플링
        if sample_rate != self.target_sample_rate:
            transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = transform(waveform)

        # 길이 맞추기
        waveform = self.pad_or_trim_waveform(waveform)

        return waveform

    def pad_or_trim_waveform(self, waveform):
        current_length = waveform.shape[1]
        if current_length > self.target_length:
            # 잘라내기
            waveform = waveform[:, :self.target_length]
        elif current_length < self.target_length:
            # 패딩
            pad_length = self.target_length - current_length
            waveform = F.pad(waveform, (0, pad_length))

        return waveform


# CNN 모델 정의
class CNNClassifier(nn.Module):
    def __init__(self, input_shape, num_label):
        super(CNNClassifier, self).__init__()
        self.norm_layer = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()

        # CNN 레이어 이후 출력 크기 계산
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape).unsqueeze(0)
            sample_output = self._forward_features(sample_input)
            flatten_size = sample_output.numel()

        self.fc1 = nn.Linear(flatten_size, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_label)

    def _forward_features(self, x):
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        x = self.norm_layer(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def main():
    # 데이터 준비
    data_dir = './dataset/'  # 데이터 폴더 경로
    # dataset = AudioFolderDataset(data_dir, transform=preprocess)
    dataset = AudioFolderDataset(root_dir=data_dir, target_sample_rate=8000, target_length=8000)

    # 시드 고정
    seed = 42
    torch.manual_seed(seed)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 모델 초기화
    input_shape = (1, 8000)  # 입력 크기를 (채널, 길이)로 설정
    num_label = len(dataset.label_map)
    model = CNNClassifier(input_shape, num_label)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 모델 훈련 및 검증
    num_epochs = 150
    best_val_accuracy = 0.0
    best_model_path = './model/model_1.pth'

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, 8000) 형태로 변환
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

        # 검증 루프
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.unsqueeze(1)  # (batch_size, 1, 8000) 형태로 변환
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(all_labels, all_preds)
        print(f'Validation Accuracy: {val_accuracy}')

        # 최적 모델 저장
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved Best Model with Accuracy: {val_accuracy}')

    best_model_path = './model/model_1.pth'

    # 모델 초기화
    input_shape = (1, 8000)  # 입력 크기를 (채널, 길이)로 설정
    num_label = len(dataset.label_map)
    print(dataset.label_map)
    model = CNNClassifier(input_shape, num_label)
    # 모델 평가
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, 8000) 형태로 변환
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy}')


if __name__ == "__main__":
    main()
