from moviepy.editor import VideoFileClip
import torch
import torchaudio
import torch.nn.functional as F


def preprocess_inference(waveform, new_sample_rate=8000):
    transform = torchaudio.transforms.Resample(orig_freq=waveform.shape[1], new_freq=new_sample_rate)
    return transform(waveform)


def predict_audio_class(model, file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    waveform = preprocess_inference(waveform)
    waveform = waveform.unsqueeze(0)
    waveform = waveform.unsqueeze(1)

    with torch.no_grad():
        outputs = model(waveform)
        _, predicted = torch.max(outputs, 1)

    label_map = {0: 'Belly_pain', 1: 'Cold_hot', 2: 'Discomfort', 3: 'Don’t_know', 4: 'Hungry', 5: 'Lonely',
                 6: 'Needs_to_burp', 7: 'Scared', 8: 'Tired'}
    predicted_class = label_map[predicted.item()]
    return predicted_class
