# teachable_app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import time
import tempfile
import os

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Teachable Machine Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ส่วนหัวแอปพลิเคชัน
st.title("🤖 Teachable Machine Image Classifier")
st.markdown("""
ใช้โมเดลจาก [Teachable Machine](https://teachablemachine.withgoogle.com/) เพื่อจำแนกภาพ
อัปโหลดไฟล์โมเดล (.h5) และไฟล์ป้ายกำกับ (.txt) ของคุณ แล้วทดสอบกับภาพ!
""")

# ==================== ส่วนแก้ไขปัญหา DepthwiseConv2D ====================
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    """แก้ไขปัญหา 'groups' parameter ในโมเดลจาก Teachable Machine"""
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# ==================== ฟังก์ชันเตรียมภาพ ====================
def prepare_image(image, target_size=(224, 224)):
    """
    เตรียมภาพสำหรับป้อนเข้าโมเดล
    - ปรับขนาด
    - ครอปส่วนกลาง
    - ปรับค่าสี
    """
    # แปลงเป็น RGB (ป้องกันปัญหา RGBA)
    img = image.convert('RGB')
    
    # ปรับขนาดโดยคงอัตราส่วนและครอปส่วนกลาง
    img = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS)
    
    # แปลงเป็น array และปรับค่าสี
    img_array = np.asarray(img)
    normalized_array = (img_array.astype(np.float32) / 127.5) - 1
    
    return img, normalized_array

# ==================== ฟังก์ชันโหลดโมเดล ====================
@st.cache_resource
def load_tm_model(model_file, labels_file):
    """โหลดโมเดลและป้ายกำกับ"""
    try:
        # สร้างไฟล์ชั่วคราวสำหรับโมเดล
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as model_temp:
            model_temp.write(model_file.getvalue())
            model_path = model_temp.name
        
        # สร้างไฟล์ชั่วคราวสำหรับป้ายกำกับ
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as labels_temp:
            labels_temp.write(labels_file.getvalue())
            labels_path = labels_temp.name
        
        # โหลดโมเดล
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D}
        )
        
        # โหลดป้ายกำกับ
        with open(labels_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f.readlines()]
        
        # ลบไฟล์ชั่วคราว
        os.unlink(model_path)
        os.unlink(labels_path)
        
        return model, class_names
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
        return None, None

# ==================== ฟังก์ชันทำนาย ====================
def predict_image(model, image_array, class_names):
    """ทำนายภาพและส่งกลับผลลัพธ์"""
    # สร้าง input tensor
    input_tensor = np.expand_dims(image_array, axis=0)
    
    # ทำนาย
    predictions = model.predict(input_tensor)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]
    
    # หาค่า confidence สูงสุด 3 อันดับ
    top_indices = np.argsort(predictions[0])[::-1][:3]
    top_results = [
        (class_names[i], float(predictions[0][i])) 
        for i in top_indices
    ]
    
    return class_names[predicted_index], confidence, top_results

# ==================== ส่วนติดต่อผู้ใช้ ====================
def main():
    # แถบด้านข้างสำหรับอัปโหลดโมเดล
    with st.sidebar:
        st.header("⚙️ การตั้งค่าโมเดล")
        model_file = st.file_uploader("อัปโหลดโมเดล (keras_Model.h5)", type=["h5"])
        labels_file = st.file_uploader("อัปโหลดไฟล์ป้ายกำกับ (labels.txt)", type=["txt"])
        
        st.markdown("---")
        st.markdown("### คำแนะนำการใช้งาน")
        st.markdown("""
        1. อัปโหลดไฟล์โมเดล (.h5) และไฟล์ป้ายกำกับ (.txt) จาก Teachable Machine
        2. อัปโหลดภาพที่ต้องการทดสอบ
        3. รอผลการจำแนกภาพ
        """)
        
        st.markdown("### ข้อมูลโมเดล")
        if model_file and labels_file:
            st.success("โหลดไฟล์โมเดลและป้ายกำกับสำเร็จ!")
        else:
            st.info("กรุณาอัปโหลดไฟล์โมเดลและป้ายกำกับ")
    
    # คอลัมน์หลัก
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.header("📤 อัปโหลดภาพ")
        uploaded_image = st.file_uploader(
            "เลือกภาพที่ต้องการทดสอบ", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="image_uploader"
        )
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)
    
    with col2:
        st.header("📊 ผลลัพธ์การจำแนก")
        
        if model_file and labels_file and uploaded_image:
            # โหลดโมเดล
            with st.spinner("กำลังโหลดโมเดล..."):
                model, class_names = load_tm_model(model_file, labels_file)
            
            if model is not None:
                # เตรียมภาพ
                with st.spinner("กำลังประมวลผลภาพ..."):
                    display_img, processed_img = prepare_image(image)
                
                # ทำนาย
                with st.spinner("กำลังวิเคราะห์ภาพ..."):
                    start_time = time.time()
                    class_name, confidence, top_results = predict_image(model, processed_img, class_names)
                    end_time = time.time()
                
                # แสดงผลลัพธ์
                st.success("การวิเคราะห์เสร็จสมบูรณ์!")
                
                # แสดงภาพที่ปรับขนาดแล้ว
                st.subheader("ภาพที่เตรียมสำหรับโมเดล (224x224)")
                st.image(display_img, use_column_width=True)
                
                # แสดงผลลัพธ์หลัก
                st.subheader("ผลการจำแนก")
                # แยกรหัสคลาสและชื่อคลาส (ถ้ามี)
                if ' ' in class_name:
                    class_id, class_label = class_name.split(' ', 1)
                else:
                    class_id = ""
                    class_label = class_name
                    
                st.markdown(f"### คลาส: **{class_label}**")
                st.markdown(f"### ความมั่นใจ: **{confidence*100:.2f}%**")
                st.markdown(f"เวลาในการประมวลผล: **{end_time - start_time:.2f} วินาที**")
                
                # แผนภูมิความมั่นใจ
                st.subheader("ความมั่นใจใน 3 อันดับแรก")
                # เตรียมข้อมูลสำหรับแผนภูมิ
                chart_labels = []
                chart_scores = []
                for name, score in top_results:
                    if ' ' in name:
                        _, label_part = name.split(' ', 1)
                    else:
                        label_part = name
                    chart_labels.append(f"{label_part} ({score*100:.1f}%)")
                    chart_scores.append(score)
                
                st.bar_chart(
                    data=dict(zip(chart_labels, chart_scores)),
                    use_container_width=True
                )
                
                # แสดงผลลัพธ์แบบตาราง
                st.subheader("ผลลัพธ์แบบละเอียด")
                result_table = []
                for i, (name, score) in enumerate(top_results):
                    if ' ' in name:
                        class_id, class_label = name.split(' ', 1)
                    else:
                        class_id = ""
                        class_label = name
                    result_table.append({
                        "อันดับ": i+1,
                        "รหัสคลาส": class_id,
                        "ชื่อคลาส": class_label,
                        "ความมั่นใจ": f"{score*100:.2f}%"
                    })
                
                st.table(result_table)
                
            else:
                st.error("ไม่สามารถโหลดโมเดลได้ กรุณาตรวจสอบไฟล์อีกครั้ง")
        else:
            st.info("กรุณาอัปโหลดโมเดล, ป้ายกำกับ และภาพเพื่อเริ่มการจำแนก")

# รันแอปพลิเคชัน
if __name__ == "__main__":
    main()pip install -r requirements.txt