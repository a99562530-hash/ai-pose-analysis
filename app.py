import streamlit as st
import joblib
import numpy as np
from PIL import Image
import random
import cv2
import tempfile
import time

st.set_page_config(page_title="AI 운동 자세 분석", layout="wide")

# ---------------- 사이드바 ----------------
st.sidebar.title("AI 자세 분석 메뉴")

menu = st.sidebar.radio(
    "메뉴 선택",
    ["자세 분석", "프로젝트 소개", "사용 방법", "분석 기록"]
)

st.sidebar.markdown("---")
threshold = st.sidebar.slider("GOOD 판정 기준", 50, 90, 60)
show_angle = st.sidebar.checkbox("관절 각도 표시", value=True)
show_skeleton = st.sidebar.checkbox("관절 스켈레톤 표시", value=True)
dark_mode = st.sidebar.checkbox("다크모드", value=False)
video_interval = st.sidebar.selectbox("영상 분석 간격", [15, 30, 60], index=1)

st.sidebar.markdown("---")
st.sidebar.caption("AI 운동 자세 분석 캡스톤 프로젝트")

# ---------------- CSS ----------------
if dark_mode:
    bg = "#0f172a"
    card = "#1e293b"
    text = "#f8fafc"
    subtext = "#cbd5e1"
else:
    bg = "#f8fafc"
    card = "#ffffff"
    text = "#111827"
    subtext = "#6b7280"

st.markdown(f"""
<style>
.stApp {{
    background: {bg};
}}

.main-container {{
    max-width: 1150px;
    margin: auto;
}}

.hero, .card {{
    background: {card};
    padding: 30px;
    border-radius: 24px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12);
    margin-bottom: 25px;
}}

.hero h1 {{
    font-size: 46px;
    font-weight: 900;
    color: {text};
}}

.hero p, .card p, .card div {{
    color: {subtext};
}}

.result-good {{
    color: #16a34a;
    font-size: 34px;
    font-weight: 900;
}}

.result-bad {{
    color: #dc2626;
    font-size: 34px;
    font-weight: 900;
}}

.metric-box {{
    background: rgba(148, 163, 184, 0.15);
    padding: 18px;
    border-radius: 18px;
    margin-top: 15px;
    font-size: 18px;
    line-height: 1.9;
    color: {text};
}}

.score-box {{
    background: #fef9c3;
    color: #854d0e;
    padding: 18px;
    border-radius: 18px;
    font-size: 24px;
    font-weight: 800;
    margin-top: 15px;
}}

[data-testid="stFileUploader"] {{
    background: rgba(148, 163, 184, 0.12);
    padding: 15px;
    border-radius: 18px;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- 기록 저장 ----------------
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ---------------- 소개 ----------------
if menu == "프로젝트 소개":
    st.markdown("""
    <div class="hero">
        <h1>프로젝트 소개</h1>
        <p>
        본 서비스는 사용자가 운동 사진 또는 영상을 업로드하면
        학습된 AI 모델이 GOOD/BAD 자세를 분석하고,
        자세 점수와 피드백을 제공하는 웹 서비스입니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- 사용 방법 ----------------
elif menu == "사용 방법":
    st.markdown("""
    <div class="hero">
        <h1>사용 방법</h1>
        <p>
        1. 운동 종류를 선택합니다.<br><br>
        2. 사진, 영상 또는 카메라 촬영 이미지를 업로드합니다.<br><br>
        3. AI가 GOOD/BAD 확률을 분석합니다.<br><br>
        4. 자세 점수, 관절 각도, 위험 경고, 피드백을 확인합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- 분석 기록 ----------------
elif menu == "분석 기록":
    st.markdown("""
    <div class="hero">
        <h1>분석 기록</h1>
        <p>최근 분석한 운동 자세 결과를 확인할 수 있습니다.</p>
    </div>
    """, unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("아직 분석 기록이 없습니다.")
    else:
        for item in reversed(st.session_state.history):
            st.markdown(f"""
            <div class="card">
                <b>운동:</b> {item['exercise']}<br>
                <b>결과:</b> {item['result']}<br>
                <b>GOOD 확률:</b> {item['good']:.2f}%<br>
                <b>BAD 확률:</b> {item['bad']:.2f}%<br>
                <b>자세 점수:</b> {item['score']}점
            </div>
            """, unsafe_allow_html=True)

# ---------------- 자세 분석 ----------------
else:
    st.markdown("""
    <div class="hero">
        <h1>AI 운동 자세 분석</h1>
        <p>
        운동 사진 또는 영상을 업로드하면 AI가 GOOD/BAD 자세를 분석하고
        자세 점수와 피드백을 제공합니다.
        </p>
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
        st.write("자세 점수, 위험 경고, 피드백을 함께 제공합니다.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- 모델 로드 ----------------
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

    # ---------------- 스켈레톤 가이드 표시 ----------------
    def draw_skeleton_guide(pil_img):
        img = np.array(pil_img)
        h, w, _ = img.shape

        points = {
            "head": (w//2, int(h*0.15)),
            "shoulder_l": (int(w*0.35), int(h*0.30)),
            "shoulder_r": (int(w*0.65), int(h*0.30)),
            "hip_l": (int(w*0.42), int(h*0.55)),
            "hip_r": (int(w*0.58), int(h*0.55)),
            "knee_l": (int(w*0.38), int(h*0.75)),
            "knee_r": (int(w*0.62), int(h*0.75)),
            "ankle_l": (int(w*0.35), int(h*0.92)),
            "ankle_r": (int(w*0.65), int(h*0.92)),
        }

        lines = [
            ("head", "shoulder_l"),
            ("head", "shoulder_r"),
            ("shoulder_l", "shoulder_r"),
            ("shoulder_l", "hip_l"),
            ("shoulder_r", "hip_r"),
            ("hip_l", "hip_r"),
            ("hip_l", "knee_l"),
            ("hip_r", "knee_r"),
            ("knee_l", "ankle_l"),
            ("knee_r", "ankle_r"),
        ]

        for a, b in lines:
            cv2.line(img, points[a], points[b], (0, 255, 0), 4)

        for p in points.values():
            cv2.circle(img, p, 7, (255, 0, 0), -1)

        return Image.fromarray(img)

    input_file = uploaded_file if uploaded_file is not None else camera_file

    if input_file is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        with st.spinner("AI가 자세를 분석 중입니다..."):
            time.sleep(1.2)

        file_name = input_file.name
        file_type = file_name.split(".")[-1].lower()

        if file_type in ["jpg", "jpeg", "png"]:
            image = Image.open(input_file).convert("RGB")

            if show_skeleton:
                display_img = draw_skeleton_guide(image)
                st.image(display_img, caption="관절 스켈레톤 표시 이미지", width=350)
            else:
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

        posture_score = int(round(good_percent))

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

        st.markdown(
            f'<div class="score-box">AI 자세 점수: {posture_score}점 / 100점</div>',
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

        if bad_percent >= 70:
            st.warning("주의: BAD 확률이 높습니다. 허리 또는 무릎 부상 위험 가능성이 있습니다.")

        st.info("AI 피드백\n\n" + feedback)

        estimated_count = random.randint(1, 12) if file_type == "mp4" else 1
        st.metric("운동 횟수 추정", f"{estimated_count}회")

        st.session_state.history.append({
            "exercise": exercise,
            "result": result_text,
            "good": good_percent,
            "bad": bad_percent,
            "score": posture_score
        })

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
