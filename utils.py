# 모듈 설치:
from moviepy.editor import VideoFileClip
import torch
import torchaudio
import torch.nn.functional as F

# 전처리 함수 재정의 (Inference 시 필요)
def preprocess_inference(waveform, new_sample_rate=8000):
    transform = torchaudio.transforms.Resample(orig_freq=waveform.shape[1], new_freq=new_sample_rate)
    return transform(waveform)

def extract_audio_from_video(video_file_path, audio_file_path): ## mp4를 wav로 저장
    # mp4 등 비디오 파일 불러오기
    video = VideoFileClip(video_file_path)

    # 오디오를 추출하여 mp3 파일로 저장
    video.audio.write_audiofile(audio_file_path, codec='pcm_s16le')

def predict_audio_class(model, file_path, label_map):
    # 오디오 파일 로드
    waveform, sample_rate = torchaudio.load(file_path)

    # 전처리 수행
    waveform = preprocess_inference(waveform)
    waveform = waveform.unsqueeze(0)  # 배치 차원 추가
    waveform = waveform.unsqueeze(1)  # 채널 차원 추가 (batch_size, 1, sample_length)

    # 모델을 사용하여 예측 수행
    with torch.no_grad():
        outputs = model(waveform)
        _, predicted = torch.max(outputs, 1)

    # 예측된 클래스 반환
    print(predicted)
    predicted_class = label_map[predicted.item()]
    return predicted_class