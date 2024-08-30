from model import CNNClassifier
from utils import extract_audio_from_video, preprocess_inference, predict_audio_class

if __name__ == '__main__':

    video_file = './mp4_folder/example.mp4'  # 변환하고 싶은 비디오 파일의 경로
    audio_file = './query_set/query_audio.wav'  # 저장할 오디오 파일의 경로, 이름 지정

    extract_audio_from_video(video_file, audio_file)

    # 모델 클래스(CNNClassifier) 및 기타 필요한 클래스와 함수는 동일하게 유지
    best_model_path = './model/model_1.pth'
    label_map = {0: 'burping', 1: 'belly_pain', 2: 'discomfort', 3: 'hungry', 4: 'tired'}
    # 모델 초기화 및 가중치 로드
    input_shape = (1, 8000)  # 입력 크기를 (채널, 길이)로 설정
    num_label = len(label_map)  # 학습 때 사용했던 label_map의 길이
    # num_label = 9
    model = CNNClassifier(input_shape, num_label)
    model.load_state_dict(torch.load(best_model_path))  # 학습된 가중치 로드
    model.eval()  # 모델을 평가 모드로 전환

    # 예측 수행
    predicted_class = predict_audio_class(audio_file)
    # print(f'The predic`ted class for the input audio file is: {predicted_class}')
    return predicted_class