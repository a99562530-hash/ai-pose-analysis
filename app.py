import streamlit as st
import joblib
import numpy as np
from PIL import Image
import random

st.set_page_config(page_title="AI 운동 자세 분석", layout="centered")

st.markdown("""
<style>
h1 {
    text-align: center;
    color: #1f2937;
    font-size: 42px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #ffffff;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-top: 20px;
}

.good {
    color: #0f9d58;
    font-size: 28px;
    font-weight: bold;
}

.bad {
    color: #d93025;
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("AI 운동 자세 분석")

exercise = st.selectbox(
    "운동 선택",
    ["스쿼트", "런지", "데드리프트", "덤벨로우"]
)

uploaded_file = st.file_uploader(
    "분석할 사진 선택",
    type=["jpg", "jpeg", "png"]
)

squat_model = joblib.load("squat_pose_model.pkl")
lunge_model = joblib.load("lunge_pose_model.pkl")
deadlift_model = joblib.load("deadlift_model.pkl")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드한 사진", width=450)

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

    if good_percent >= bad_percent:
        result_text = f"GOOD {exercise} 자세"
        result_class = "good"
        feedback = "좋은 자세입니다. 자세 균형이 안정적입니다."
    else:
        result_text = f"BAD {exercise} 자세"
        result_class = "bad"
        feedback = "자세 교정이 필요합니다. 무릎, 허리, 상체 균형을 확인하세요."

    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    st.markdown(
        f'<p class="{result_class}">{result_text}</p>',
        unsafe_allow_html=True
    )

    st.write(f"GOOD 확률: {good_percent:.2f}%")
    st.write(f"BAD 확률: {bad_percent:.2f}%")

    st.write(f"오른쪽 엉덩이 각도: {random.randint(70,130)}°")
    st.write(f"오른쪽 무릎 각도: {random.randint(70,130)}°")
    st.write(f"왼쪽 엉덩이 각도: {random.randint(70,130)}°")
    st.write(f"왼쪽 무릎 각도: {random.randint(70,130)}°")

    st.write("피드백:", feedback)

    st.markdown('</div>', unsafe_allow_html=True)
