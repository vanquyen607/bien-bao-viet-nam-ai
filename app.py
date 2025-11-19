# app.py - BIỂN BÁO VIỆT NAM AI 2025 - VIDEO + WEBCAM HIỆN TÊN + % SIÊU RÕ
import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# ==================== TÌM MODEL ====================
def find_model():
    paths = ["best.pt", "weights/best.pt", "content/runs/vietnamese_traffic_signs/weights/best.pt"]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

model_path = find_model()
if not model_path:
    st.error("Không tìm thấy file best.pt!")
    st.stop()

@st.cache_resource
def load_model():
    return YOLO(model_path)

model = load_model()

# ==================== NHÓM BIỂN BÁO ====================
SIGN_GROUPS = {
    "Cấm": ["No ", "No Entry", "No Left Turn", "No Right Turn", "No U-Turn", "No Parking", "No Stopping"],
    "Hạn chế tốc độ": ["Speed limit"],
    "Nguy hiểm": ["Danger", "Uneven Road", "Children Present", "Obstacle"],
    "Chỉ dẫn": ["Right Turn Only", "Left Turn", "Roundabout", "Bus Stop", "Pedestrian Crossing"],
    "Khác": []
}

def get_group(name):
    for group, keywords in SIGN_GROUPS.items():
        if any(k in name for k in keywords):
            return group
    return "Khác"

if "all_results" not in st.session_state:
    st.session_state.all_results = []

# ==================== BACKGROUND + CSS ====================
background_url = "https://images.unsplash.com/photo-1501829634390-7c98deb33171?q=80&w=1170&auto=format&fit=crop"

st.markdown(f"""
<style>
    .stApp {{background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url("{background_url}"); background-size: cover; background-attachment: fixed;}}
    .sidebar .sidebar-content {{background: rgba(0, 60, 120, 0.98);}}
    .stButton>button {{background: #00ccff; color: white; font-weight: bold; border-radius: 15px; height: 3.5em; font-size: 1.3rem;}}
    h1, h2, h3, p, li, span, div {{color: white !important; text-shadow: 1px 1px 3px black;}}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Biển Báo VN AI", page_icon="🇻🇳", layout="wide")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/21/Flag_of_Vietnam.svg", width=100)
    st.markdown("<h2 style='color:#00ccff;'>BIỂN BÁO VN AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:white;'><b>mAP50 = 98%</b></p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Chọn chức năng", ["📸 Ảnh", "🎥 Video", "📷 Webcam Realtime", "📊 Phân tích EDA"], label_visibility="collapsed")

st.markdown("<h1 style='text-align:center; color:#00ccff;'>🚦 NHẬN DIỆN BIỂN BÁO GIAO THÔNG VIỆT NAM</h1>", unsafe_allow_html=True)

# ==================== CÁC TRANG ====================
if page == "📸 Ảnh":
    st.markdown("### Upload ảnh biển báo")
    uploaded = st.file_uploader("", type=["jpg","png","jpeg","bmp","webp"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        with col1: st.image(img, caption="Ảnh gốc", use_container_width=True)
        with col2:
            with st.spinner("Đang nhận diện..."):
                res = model(img, conf=0.5)[0]
                annotated = res.plot(line_width=6, font_size=2.0, pil=True, labels=True, conf=True)
                st.image(annotated, caption="Kết quả AI", use_container_width=True)
                for box in res.boxes:
                    name = res.names[int(box.cls)]
                    conf = box.conf.item()
                    st.markdown(f"✅ **{name}** – **{conf:.1%}**")
                    st.session_state.all_results.append({"Biển báo": name, "Độ tin cậy": round(conf, 3)})

elif page == "🎥 Video":
    st.markdown("### Upload video đường phố")
    vid = st.file_uploader("", type=["mp4","avi","mov","mkv"])
    if vid:
        st.video(vid)
        if st.button("BẮT ĐẦU NHẬN DIỆN VIDEO", type="primary", use_container_width=True):
            tfile = "temp_video.mp4"
            with open(tfile, "wb") as f: f.write(vid.read())
            cap = cv2.VideoCapture(tfile)
            ph = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                res = model(frame, conf=0.5)[0]
                # ĐÃ SỬA ĐỂ HIỆN TÊN + % TRÊN VIDEO NHƯ WEBCAM
                annotated = res.plot(
                    line_width=6,
                    font_size=20,
                    pil=True,
                    labels=True,
                    conf=True
                )
                ph.image(annotated, channels="BGR", use_container_width=True, clamp=True)
                for box in res.boxes:
                    name = res.names[int(box.cls)]
                    conf = box.conf.item()
                    st.session_state.all_results.append({"Biển báo": name, "Độ tin cậy": round(conf, 3)})
            cap.release()
            os.remove(tfile)
            st.success("Xử lý video hoàn tất!")

elif page == "📷 Webcam Realtime":
    st.markdown("### WEBCAM REALTIME – TÊN + % HIỆN SIÊU RÕ")
    run = st.checkbox("BẬT CAMERA", value=True)
    frame_window = st.image([])
    cap = cv2.VideoCapture(0)

    if run:
        while run:
            ret, frame = cap.read()
            if not ret: break
            res = model(frame, conf=0.5)[0]
            annotated = res.plot(
                line_width=6,
                font_size=20,
                pil=True,
                labels=True,
                conf=True
            )
            frame_window.image(annotated, channels="BGR", use_container_width=True, clamp=True)
            for box in res.boxes:
                name = res.names[int(box.cls)]
                conf = box.conf.item()
                st.session_state.all_results.append({"Biển báo": name, "Độ tin cậy": round(conf, 3)})
    else:
        cap.release()

else:  # ==================== EDA 6 BIỂU ĐỒ ====================
    if not st.session_state.all_results:
        st.info("Chưa có dữ liệu!")
    else:
        df = pd.DataFrame(st.session_state.all_results)
        df["Nhóm"] = df["Biển báo"].apply(get_group)

        st.markdown("<h2 style='color:#00ccff; text-align:center;'>6 BIỂU ĐỒ PHÂN TÍCH CHI TIẾT</h2>", unsafe_allow_html=True)

        plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})

        fig = plt.figure(figsize=(8,6), facecolor='#0e1117')
        top10 = df["Biển báo"].value_counts().head(10)
        sns.barplot(y=top10.index, x=top10.values, palette="Blues_r")
        plt.title("1. Top 10 biển báo phổ biến", fontsize=16, fontweight="bold")
        plt.xlabel("Số lần"); plt.ylabel("")
        plt.gca().set_facecolor('#0e1117')
        st.pyplot(fig); plt.clf()

        fig = plt.figure(figsize=(8,6), facecolor='#0e1117')
        sns.histplot(df["Độ tin cậy"], bins=20, kde=True, color="#00ccff", edgecolor="white")
        plt.title("2. Phân bố độ tin cậy", fontsize=16, fontweight="bold")
        plt.gca().set_facecolor('#0e1117')
        st.pyplot(fig); plt.clf()

        fig = plt.figure(figsize=(8,6), facecolor='#0e1117')
        group = df["Nhóm"].value_counts()
        colors = ["#00ccff","#0099cc","#0077aa","#005588","#003366"]
        plt.pie(group.values, labels=group.index, autopct='%1.1f%%', colors=colors, textprops={'color':'white','weight':'bold','fontsize':12})
        plt.title("3. Tỷ lệ theo nhóm", fontsize=16, fontweight="bold")
        st.pyplot(fig); plt.clf()

        fig = plt.figure(figsize=(8,6), facecolor='#0e1117')
        sns.boxplot(x="Nhóm", y="Độ tin cậy", data=df, palette="Blues")
        plt.title("4. Độ tin cậy theo nhóm", fontsize=16, fontweight="bold")
        plt.gca().set_facecolor('#0e1117'); plt.xticks(rotation=15)
        st.pyplot(fig); plt.clf()

        fig = plt.figure(figsize=(8,6), facecolor='#0e1117')
        recent = df.tail(50)
        plt.plot(range(len(recent)), recent["Độ tin cậy"], marker='o', color="#00ccff", linewidth=3)
        plt.title("5. Độ tin cậy 50 lần gần nhất", fontsize=16, fontweight="bold")
        plt.gca().set_facecolor('#0e1117')
        st.pyplot(fig); plt.clf()

        fig = plt.figure(figsize=(8,6), facecolor='#0e1117')
        df["Mức"] = pd.cut(df["Độ tin cậy"], bins=[0,0.5,0.7,0.9,1.0], labels=["<50%","50-70%","70-90%","90-100%"])
        sns.countplot(x="Mức", data=df, palette="Blues_r")
        plt.title("6. Phân bố theo mức độ tin cậy", fontsize=16, fontweight="bold")
        plt.gca().set_facecolor('#0e1117')
        st.pyplot(fig); plt.clf()

        if st.button("XÓA DỮ LIỆU"):
            st.session_state.all_results = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#00ccff; font-size:1.5rem;'><b>Made in Vietnam with ❤️ – 2025</b></p>", unsafe_allow_html=True)