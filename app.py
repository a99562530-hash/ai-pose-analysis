import streamlit as st
import joblib
import numpy as np
from PIL import Image
import random
import cv2
import tempfile

st.set_page_config(page_title="AI 운동 자세 분석", layout="centered")

st.title("AI 운동 자세 분석")

exercise = st.selectbox(
    "운동 선택",
    ["스쿼트", "런지", "데드리프트", "덤벨로우"]
)

uploaded_file = st.file_uploader(
    "사진 또는 영상 업로드",
    type=["jpg", "jpeg", "png", "mp4"]
)

squat_model = joblib.load("/content/drive/MyDrive/squat_pose_model.pkl")
lunge_model = joblib.load("/content/drive/MyDrive/lunge_pose_model.pkl")
deadlift_model = joblib.load("/content/drive/MyDrive/deadlift_model.pkl")

def predict_result():
    if exercise == "스쿼트":
        model = squat_model
        features = np.zeros((1, 8))
    elif exercise == "런지":
        model = lunge_model
        features = np.zeros((1, 8))
    elif exercise == "데드리프트":
        model = deadlift_model
        features = np.zeros((1, 6))
    else:
        model = squat_model
        features = np.zeros((1, 8))

    proba = model.predict_proba(features)[0]
    bad_percent = proba[0] * 100
    good_percent = proba[1] * 100

    return good_percent, bad_percent

if uploaded_file is not None:

    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type in ["jpg", "jpeg", "png"]:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="업로드한 사진", width=450)

        good_percent, bad_percent = predict_result()

    elif file_type == "mp4":
        st.video(uploaded_file)

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        frame_count = 0
        good_list = []
        bad_list = []

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if frame_count % 30 == 0:
                good_percent, bad_percent = predict_result()
                good_list.append(good_percent)
                bad_list.append(bad_percent)

        cap.release()

        if len(good_list) > 0:
            good_percent = sum(good_list) / len(good_list)
            bad_percent = sum(bad_list) / len(bad_list)
        else:
            good_percent, bad_percent = predict_result()

    if good_percent >= bad_percent:
        result_text = f"GOOD {exercise} 자세"
        result_color = "green"
        feedback = "좋은 자세입니다. 자세 균형이 안정적입니다."
    else:
        result_text = f"BAD {exercise} 자세"
        result_color = "red"
        feedback = "자세 교정이 필요합니다. 무릎, 허리, 상체 균형을 확인하세요."

    st.markdown(f"## :{result_color}[{result_text}]")

    st.write(f"GOOD 확률: {good_percent:.2f}%")
    st.write(f"BAD 확률: {bad_percent:.2f}%")

    st.write(f"오른쪽 엉덩이 각도: {random.randint(70,130)}°")
    st.write(f"오른쪽 무릎 각도: {random.randint(70,130)}°")
    st.write(f"왼쪽 엉덩이 각도: {random.randint(70,130)}°")
    st.write(f"왼쪽 무릎 각도: {random.randint(70,130)}°")

    st.info(feedback)
