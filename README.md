# 아기 울음소리 분류모델 구축

## 데이터 셋
* data source: [donateacry-corpus](https://github.com/gveres/donateacry-corpus#donateacry-corpus), [실제 사용한 데이터 셋](https://github.com/martha92/babycry?tab=readme-ov-file)
* 데이터 셋 설명:
  1. 오디오 파일에는 0세에서 2세 사이의 영유아 울음 소리가 포함되어 있습니다.
  2. 해당 울음 소리의 원인에 대한 태그 정보가 파일 이름에 인코딩되어 있습니다.
  3. 모든 파일은 Python 오디오 라이브러리(librosa, Wave, SoundFile)로 쉽게 읽고 변환할 수 있도록 WAV 파일 형식으로 변환되었습니다.
 
* 데이터 셋 구성

|태그|클래스/울음의 이유|최종 사용한 오디오 파일 수|
|------|---|---|
|hu|Hungry|50|
|bu|Needs to burp|21|
|bp|Belly pain|24|
|dc|Discomfort|42|
|ti|Tired|56|
|lo|Lonely|44|
|ch|Cold/hot|28|
|sc|Scared|20|
|dk|Don’t know|58|

## 모델 학습 환경
* Jupyter notebook

## 모델 학습 및 추론
0. ```pip install -r requirements.txt```
1. jupyter notebook 환경에서 데이터 셋을 구축 후 모든 셀 실행.
2. ```python inference.py```
