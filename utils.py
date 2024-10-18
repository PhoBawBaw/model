import torch
import torchaudio
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram


def preprocess_audio(file_path, target_sample_rate=8000, target_length=56000):
    waveform, sample_rate = torchaudio.load(file_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resample(waveform)

    current_length = waveform.shape[1]
    if current_length > target_length:
        waveform = waveform[:, :target_length]
    elif current_length < target_length:
        pad_length = target_length - current_length
        waveform = F.pad(waveform, (0, pad_length))

    mel_spectrogram = MelSpectrogram(sample_rate=target_sample_rate, n_mels=64)
    mel_spec = mel_spectrogram(waveform)

    return mel_spec


def predict_audio_class(model, file_path):
    mel_spec = preprocess_audio(file_path)

    mel_spec = mel_spec.unsqueeze(0)

    with torch.no_grad():
        outputs = model(mel_spec)
        _, predicted_label = torch.max(outputs, 1)

    label_map = {0: 'Belly_pain', 1: 'Cold_hot', 2: 'Discomfort', 3: 'Don’t_know', 4: 'Hungry', 5: 'Lonely',
                 6: 'Needs_to_burp', 7: 'Scared', 8: 'Tired'}

    predicted_class = label_map[predicted_label.item()]
    return predicted_class
