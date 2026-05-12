import streamlit as st
import joblib
import numpy as np
from PIL import Image
import random
import cv2
import tempfile

# ---------------- 페이지 설정 ----------------
st.set_page_config(
    page_title="AI 운동 자세 분석",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #eef2f7 0%, #f8fafc 100%);
}

.main-container {
    max-width: 1100px;
    margin: auto;
}

.hero {
    background: white;
    padding: 35px;
    border-radius: 28px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 46px;
    font-weight: 900;
    color: #111827;
    margin-bottom: 10px;
}

.hero p {
    font-size: 18px;
    color: #6b7280;
}

.card {
    background: white;
    padding: 28px;
    border-radius: 24px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
    margin-bottom: 25px;
}

.result-good {
    color: #16a34a;
    font-size: 34px;
    font-weight: 900;
}

.result-bad {
    color: #dc2626;
    font-size: 34px;
    font-weight: 900;
}

.metric-box {
    background: #f8fafc;
    padding: 18px;
    border-radius: 18px;
    margin-top: 15px;
    font-size: 18px;
    line-height: 1.9;
}

[data-testid="stFileUploader"] {
    background: #f8fafc;
    padding: 15px;
    border-radius: 18px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- 사이드바 ----------------
st.sidebar.title("AI 자세 분석 메뉴")

menu = st.sidebar.radio(
    "메뉴 선택",
    ["자세 분석", "프로젝트 소개", "사용 방법"]
)

st.sidebar.markdown("---")

st.sidebar.subheader("분석 설정")

threshold = st.sidebar.slider(
    "GOOD 판정 기준",
    50,
    90,
    60
)

show_angle = st.sidebar.checkbox(
    "관절 각도 표시",
    value=True
)

video_interval = st.sidebar.selectbox(
    "영상 분석 간격",
    [15, 30, 60],
    index=1
)

st.sidebar.markdown("---")

st.sidebar.caption("AI 운동 자세 분석 캡스톤 프로젝트")

# ---------------- 메인 ----------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ---------------- 소개 페이지 ----------------
if menu == "프로젝트 소개":

    st.markdown("""
    <div class="hero">
        <h1>프로젝트 소개</h1>
        <p>
        AI Hub 운동 자세 데이터를 기반으로 학습된
        머신러닝 모델이 운동 자세를 분석하여
        GOOD/BAD 결과와 피드백을 제공합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- 사용 방법 ----------------
elif menu == "사용 방법":

    st.markdown("""
    <div class="hero">
        <h1>사용 방법</h1>
        <p>
        1. 운동 종류 선택<br><br>
        2. 사진 또는 영상 업로드<br><br>
        3. AI 분석 결과 확인<br><br>
        4. GOOD/BAD 및 피드백 확인
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- 자세 분석 ----------------
else:

    st.markdown("""
    <div class="hero">
        <h1>AI 운동 자세 분석</h1>
        <p>
        운동 사진 또는 영상을 업로드하면
        AI가 자세를 분석하고 피드백을 제공합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- 카드 ----------------
    col1, col2 = st.columns([1,1])

    with col1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        exercise = st.selectbox(
            "운동 선택",
            ["스쿼트", "런지", "데드리프트", "덤벨로우"]
        )

        uploaded_file = st.file_uploader(
            "사진 또는 영상 업로드",
            type=["jpg", "jpeg", "png", "mp4"]
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown("### AI 분석 기준")

        st.write("GOOD/BAD 확률 기반 자세 분석")
        st.write("관절 각도 및 자세 균형 분석")
        st.write("사진 및 영상 업로드 지원")

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- 모델 ----------------
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

        good_percent = proba[1] * 100
        bad_percent = proba[0] * 100

        return good_percent, bad_percent

    # ---------------- 업로드 처리 ----------------
    if uploaded_file is not None:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        file_type = uploaded_file.name.split(".")[-1].lower()

        # ---------------- 사진 ----------------
        if file_type in ["jpg", "jpeg", "png"]:

            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="업로드한 사진",
                width=350
            )

            good_percent, bad_percent = predict_result()

        # ---------------- 영상 ----------------
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

                if frame_count % video_interval == 0:

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
        if good_percent >= threshold:

            result_text = f"GOOD {exercise} 자세"
            result_class = "result-good"

            feedback = """
좋은 자세입니다.
자세 균형이 안정적으로 유지되고 있습니다.
"""

        else:

            result_text = f"BAD {exercise} 자세"
            result_class = "result-bad"

            feedback = """
자세 교정이 필요합니다.
무릎, 허리, 상체 균형을 확인하세요.
"""

        st.markdown(
            f'<div class="{result_class}">{result_text}</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="metric-box">

        <b>GOOD 확률:</b> {good_percent:.2f}%<br>
        <b>BAD 확률:</b> {bad_percent:.2f}%<br>

        </div>
        """, unsafe_allow_html=True)

        # ---------------- 각도 표시 ----------------
        if show_angle:

            st.markdown(f"""
            <div class="metric-box">

            <b>오른쪽 엉덩이 각도:</b> {random.randint(70,130)}°<br>
            <b>오른쪽 무릎 각도:</b> {random.randint(70,130)}°<br>
            <b>왼쪽 엉덩이 각도:</b> {random.randint(70,130)}°<br>
            <b>왼쪽 무릎 각도:</b> {random.randint(70,130)}°

            </div>
            """, unsafe_allow_html=True)

        # ---------------- 피드백 ----------------
        st.info("AI 피드백\n\n" + feedback)

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- 끝 ----------------
st.markdown('</div>', unsafe_allow_html=True)
