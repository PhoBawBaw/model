FROM python:3.7-slim

WORKDIR /app

COPY . .

RUN python -m venv venv

RUN ./venv/bin/pip install --upgrade pip
RUN ./venv/bin/pip install -r requirements.txt

ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV BENTOML_HOME=/bentoml
RUN mkdir -p /app/bentoml

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "1213"]