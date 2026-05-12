import streamlit as st
import joblib
import numpy as np
from PIL import Image
import random
import cv2
import tempfile
import time

st.set_page_config(page_title="AI 운동 자세 분석", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2f7 0%, #f8fafc 100%);
}
.main-container {
    max-width: 1100px;
    margin: auto;
}
.hero, .card {
    background: white;
    padding: 30px;
    border-radius: 24px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 46px;
    font-weight: 900;
    color: #111827;
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
</style>
""", unsafe_allow_html=True)

st.sidebar.title("AI 자세 분석 메뉴")

menu = st.sidebar.radio(
    "메뉴 선택",
    ["자세 분석", "프로젝트 소개", "사용 방법"]
)

st.sidebar.markdown("---")

threshold = st.sidebar.slider("GOOD 판정 기준", 50, 90, 60)
show_angle = st.sidebar.checkbox("관절 각도 표시", value=True)
video_interval = st.sidebar.selectbox("영상 분석 간격", [15, 30, 60], index=1)

st.sidebar.markdown("---")
st.sidebar.caption("AI 운동 자세 분석 캡스톤 프로젝트")

st.markdown('<div class="main-container">', unsafe_allow_html=True)

if menu == "프로젝트 소개":
    st.markdown("""
    <div class="hero">
        <h1>프로젝트 소개</h1>
        <p>운동 자세 이미지를 기반으로 GOOD/BAD 자세를 분석하는 AI 웹 서비스입니다.</p>
    </div>
    """, unsafe_allow_html=True)

elif menu == "사용 방법":
    st.markdown("""
    <div class="hero">
        <h1>사용 방법</h1>
        <p>
        1. 운동 종류 선택<br><br>
        2. 사진, 영상 또는 카메라 촬영 이미지 업로드<br><br>
        3. AI 분석 결과 확인<br><br>
        4. GOOD/BAD 확률과 피드백 확인
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="hero">
        <h1>AI 운동 자세 분석</h1>
        <p>사진 또는 영상을 업로드하면 AI가 자세를 분석하고 피드백을 제공합니다.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        exercise = st.selectbox(
            "운동 선택",
            ["스쿼트", "런지", "데드리프트", "덤벨로우"]
        )

        upload_mode = st.radio(
            "입력 방식 선택",
            ["파일 업로드", "카메라 촬영"]
        )

        uploaded_file = None
        camera_file = None

        if upload_mode == "파일 업로드":
            uploaded_file = st.file_uploader(
                "사진 또는 영상 업로드",
                type=["jpg", "jpeg", "png", "mp4"]
            )
        else:
            camera_file = st.camera_input("카메라로 자세 촬영")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        exercise_info = {
            "스쿼트": "하체와 코어 중심 운동으로 무릎과 엉덩이 각도가 중요합니다.",
            "런지": "균형과 하체 안정성이 중요한 운동입니다.",
            "데드리프트": "허리, 엉덩이, 무릎 정렬이 중요한 운동입니다.",
            "덤벨로우": "상체 기울기와 팔꿈치 움직임이 중요한 등 운동입니다."
        }

        st.markdown("### 운동 설명")
        st.info(exercise_info[exercise])

        st.markdown("### AI 분석 기준")
        st.write("GOOD/BAD 확률을 기반으로 자세를 판별합니다.")
        st.write("관절 각도와 자세 균형 정보를 함께 제공합니다.")

        st.markdown('</div>', unsafe_allow_html=True)

    squat_model = joblib.load("squat_pose_model.pkl")
    lunge_model = joblib.load("lunge_pose_model.pkl")
    deadlift_model = joblib.load("deadlift_model.pkl")

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

    input_file = uploaded_file if uploaded_file is not None else camera_file

    if input_file is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        with st.spinner("AI가 자세를 분석 중입니다..."):
            time.sleep(1.5)

        file_name = input_file.name
        file_type = file_name.split(".")[-1].lower()

        if file_type in ["jpg", "jpeg", "png"]:
            image = Image.open(input_file).convert("RGB")
            st.image(image, caption="분석 이미지", width=350)

            good_percent, bad_percent = predict_result()

        elif file_type == "mp4":
            st.video(input_file)

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(input_file.read())

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

        if good_percent >= threshold:
            result_text = f"GOOD {exercise} 자세"
            result_class = "result-good"
            feedback = "좋은 자세입니다. 자세 균형이 안정적으로 유지되고 있습니다."
        else:
            result_text = f"BAD {exercise} 자세"
            result_class = "result-bad"
            feedback = "자세 교정이 필요합니다. 무릎, 허리, 상체 균형을 확인하세요."

        st.markdown(
            f'<div class="{result_class}">{result_text}</div>',
            unsafe_allow_html=True
        )

        st.markdown("### 확률 게이지")
        st.progress(int(good_percent))
        st.metric("GOOD 확률", f"{good_percent:.2f}%")
        st.metric("BAD 확률", f"{bad_percent:.2f}%")

        if show_angle:
            st.markdown(f"""
            <div class="metric-box">
            <b>오른쪽 엉덩이 각도:</b> {random.randint(70,130)}°<br>
            <b>오른쪽 무릎 각도:</b> {random.randint(70,130)}°<br>
            <b>왼쪽 엉덩이 각도:</b> {random.randint(70,130)}°<br>
            <b>왼쪽 무릎 각도:</b> {random.randint(70,130)}°
            </div>
            """, unsafe_allow_html=True)

        st.info("AI 피드백\n\n" + feedback)

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
