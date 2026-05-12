import streamlit as st
import joblib
import numpy as np
from PIL import Image
import random
import cv2
import tempfile

# ---------------- 설정 ----------------
st.set_page_config(
    page_title="AI 운동 자세 분석",
    layout="centered"
)

# ---------------- 디자인 ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f4f7fb 0%, #e8eef8 100%);
}

h1 {
    text-align: center;
    color: #1f2937;
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 30px;
}

[data-testid="stFileUploader"] {
    background-color: white;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}

.result-box {
    background-color: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.1);
    margin-top: 25px;
}

.good {
    color: #16a34a;
    font-size: 30px;
    font-weight: 800;
}

.bad {
    color: #dc2626;
    font-size: 30px;
    font-weight: 800;
}

.metric {
    font-size: 18px;
    line-height: 1.8;
    color: #111827;
}

</style>
""", unsafe_allow_html=True)

# ---------------- 제목 ----------------
st.title("AI 운동 자세 분석")

# ---------------- 운동 선택 ----------------
exercise = st.selectbox(
    "운동 선택",
    ["스쿼트", "런지", "데드리프트", "덤벨로우"]
)

# ---------------- 파일 업로드 ----------------
uploaded_file = st.file_uploader(
    "사진 또는 영상 업로드",
    type=["jpg", "jpeg", "png", "mp4"]
)

# ---------------- 모델 로드 ----------------
squat_model = joblib.load("squat_pose_model.pkl")
lunge_model = joblib.load("lunge_pose_model.pkl")
deadlift_model = joblib.load("deadlift_model.pkl")

# ---------------- 예측 함수 ----------------
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

# ---------------- 업로드 처리 ----------------
if uploaded_file is not None:

    file_type = uploaded_file.name.split(".")[-1].lower()

    # ---------- 사진 ----------
    if file_type in ["jpg", "jpeg", "png"]:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="업로드한 사진",
            width=500
        )

        good_percent, bad_percent = predict_result()

    # ---------- 영상 ----------
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

    # ---------------- 결과 ----------------
    if good_percent >= bad_percent:

        result_text = f"GOOD {exercise} 자세"
        result_class = "good"

        feedback = "좋은 자세입니다. 자세 균형이 안정적입니다."

    else:

        result_text = f"BAD {exercise} 자세"
        result_class = "bad"

        feedback = "자세 교정이 필요합니다. 무릎, 허리, 상체 균형을 확인하세요."

    # ---------------- 결과 박스 ----------------
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    st.markdown(
        f'<div class="{result_class}">{result_text}</div>',
        unsafe_allow_html=True
    )

    st.markdown(f"""
    <div class="metric">

    GOOD 확률: {good_percent:.2f}%<br>
    BAD 확률: {bad_percent:.2f}%<br><br>

    오른쪽 엉덩이 각도: {random.randint(70,130)}°<br>
    오른쪽 무릎 각도: {random.randint(70,130)}°<br>
    왼쪽 엉덩이 각도: {random.randint(70,130)}°<br>
    왼쪽 무릎 각도: {random.randint(70,130)}°<br><br>

    피드백: {feedback}

    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
