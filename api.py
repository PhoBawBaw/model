from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import os
import asyncpg
from datetime import datetime
from model import CRNNAttentionClassifier
from utils import predict_audio_class
import torch
import subprocess

app = FastAPI()

best_model_path = './model/model_revision.pth'
num_classes = 6
model = CRNNAttentionClassifier(num_classes=num_classes)
model.load_state_dict(torch.load(best_model_path))
model.eval()


@app.post("/upload-wav/")
async def upload_wav(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        return JSONResponse(content={"error": "파일은 WAV 형식이어야 합니다."}, status_code=400)

    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    label = predict_audio_class(model, file_path)
    await save_to_db(label)

    os.remove(file_path)

    return {"predicted_label": label}


async def save_to_db(label: str):
    conn = await asyncpg.connect(
        host="",
        port="",
        user="",
        password="",
        database=""
    )

    timestamp = datetime.now()

    await conn.execute('''
    INSERT INTO public.status (datetime, crying) VALUES ($1, $2)
    ''', timestamp, label)

    await conn.close()


@app.post("/continual_learning/")
async def continual_learning():
    try:
        result = subprocess.run(
            ["python", "continual.py"],
            capture_output=True,
            text=True,
            check=True
        )
        return {"message": "모델 개선이 완료되었습니다!", "output": result.stdout}

    except subprocess.CalledProcessError as e:
        return {"message": "모델 개선 중 오류가 발생했습니다!", "error": e.stderr}

    except Exception as e:
        return {"message": "모델 개선 중 오류가 발생했습니다!", "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1213)
