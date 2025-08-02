from fastapi import FastAPI
from app.api.predict import router as predict_router
from app.core.model import model_instance
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="Teachable Machine App")

# โหลดโมเดลเมื่อแอปเริ่มทำงาน
@app.on_event("startup")
async def startup_event():
    print("กำลังโหลดโมเดล...")
    model_path = os.path.join(os.path.dirname(__file__), '../../keras_Model.h5')
    labels_path = os.path.join(os.path.dirname(__file__), '../../labels.txt')
    success, message = model_instance.load_model(model_path, labels_path)
    print(f"สถานะ: {'สำเร็จ' if success else 'ล้มเหลว'} - {message}")

# ลงทะเบียน route
app.include_router(predict_router, prefix="/api")

# เสิร์ฟไฟล์ static
app.mount("/", StaticFiles(directory="../../static", html=True), name="static")