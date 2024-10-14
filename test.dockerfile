# Python 3.7을 베이스 이미지로 사용
FROM python:3.7-slim

# 작업 디렉토리 설정
WORKDIR /app

# 현재 디렉토리의 모든 파일을 Docker 컨테이너의 /app 디렉토리에 복사
COPY . .

# Python 가상환경 생성
RUN python -m venv venv

# pip 업그레이드
RUN ./venv/bin/pip install --upgrade pip

# 가상환경의 pip를 사용하여 필요한 패키지 설치
RUN ./venv/bin/pip install -r requirements.txt

# 환경변수 설정: 가상환경 사용하도록 설정
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 컨테이너 실행 시 가상환경의 Python 사용하여 inference.py 실행
CMD ["./venv/bin/python", "api.py"]
