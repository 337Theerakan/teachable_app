import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import os

class TeachableModel:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.loaded = False
        
    def load_model(self, model_path, labels_path):
        """โหลดโมเดลและป้ายกำกับ"""
        # คลาสแก้ไขปัญหา DepthwiseConv2D
        class FixedDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop('groups', None)
                super().__init__(*args, **kwargs)
        
        try:
            print(f"กำลังโหลดโมเดลจาก: {model_path}")
            print(f"กำลังโหลดป้ายกำกับจาก: {labels_path}")
            
            # ตรวจสอบว่าไฟล์มีอยู่
            if not os.path.exists(model_path):
                return False, f"ไม่พบไฟล์โมเดล: {model_path}"
                
            if not os.path.exists(labels_path):
                return False, f"ไม่พบไฟล์ป้ายกำกับ: {labels_path}"
            
            # โหลดโมเดล
            self.model = load_model(
                model_path,
                compile=False,
                custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D}
            )
            
            # โหลดป้ายกำกับ
            with open(labels_path, "r", encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f.readlines()]
            
            self.loaded = True
            return True, "โมเดลโหลดสำเร็จ"
        except Exception as e:
            return False, f"เกิดข้อผิดพลาด: {str(e)}"
    
    def predict_image(self, image_path):
        """ทำนายภาพและส่งกลับผลลัพธ์"""
        if not self.loaded:
            return None, "ยังไม่ได้โหลดโมเดล"
        
        try:
            # เตรียมภาพ
            image = Image.open(image_path).convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            
            # สร้าง input data
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array
            
            # ทำนาย
            prediction = self.model.predict(data)
            index = np.argmax(prediction[0])
            class_name = self.class_names[index]
            confidence = prediction[0][index]
            
            # หา 3 อันดับแรก
            top_indices = np.argsort(prediction[0])[::-1][:3]
            top_results = [
                {"class": self.class_names[i], "confidence": float(prediction[0][i])}
                for i in top_indices
            ]
            
            return {
                "top_class": class_name,
                "confidence": float(confidence),
                "top_results": top_results
            }, None
        except Exception as e:
            return None, f"ข้อผิดพลาดในการทำนาย: {str(e)}"

# สร้างอินสแตนซ์โมเดลแบบ global
model_instance = TeachableModel()