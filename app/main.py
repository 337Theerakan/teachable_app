from fastapi import FastAPI
from app.api.predict import router as predict_router
from app.core.model import model_instance
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Teachable Machine App")

# โหลดโมเดลเมื่อแอปเริ่มทำงาน
@app.on_event("startup")
async def startup_event():
    print("กำลังโหลดโมเดล...")
    success, message = model_instance.load_model()
    print(f"สถานะ: {'สำเร็จ' if success else 'ล้มเหลว'} - {message}")

# ลงทะเบียน route
app.include_router(predict_router, prefix="/api")

# เสิร์ฟไฟล์ static
app.mount("/", StaticFiles(directory="static", html=True), name="static")
