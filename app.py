from __future__ import annotations

import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from predict import load_model


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "artifacts" / "food_classifier.pt"
METRICS_PATH = APP_DIR / "artifacts" / "metrics.json"
CONFIDENCE_PATH = APP_DIR / "artifacts" / "confidence.json"
CHARACTER_PATH = APP_DIR / "assets" / "character.png"

CALORIES_BY_CATEGORY = {"밥": 300, "국": 120, "반찬": 180, "후식": 100}
CATEGORY_ICON = {"밥": "🍚", "국": "🍲", "반찬": "🍱", "후식": "🍎"}
TODAYS_MESSAGE = {
    "밥": "든든한 한 끼가 오늘의 좋은 에너지가 되어줄 거예요.",
    "국": "따뜻한 한 그릇처럼 편안한 하루 보내세요.",
    "반찬": "다양한 맛처럼 오늘도 즐거운 일이 가득하길 바라요.",
    "후식": "달콤한 마무리처럼 기분 좋은 하루 보내세요.",
}

MESSAGE_OPENINGS = (
    "오늘의 한 끼가 든든한 힘이 되어",
    "맛있는 순간의 기분 좋은 에너지가",
    "정성스러운 한 접시의 따뜻함이",
    "잘 챙겨 먹은 오늘의 뿌듯함이",
    "향긋한 음식이 전해 준 즐거움이",
    "천천히 음미한 한 끼의 여유가",
    "좋아하는 음식을 만난 설렘이",
    "균형 있게 채운 한 끼의 힘이",
    "따뜻한 식사로 얻은 편안함이",
    "맛있게 채운 오늘의 행복이",
    "소중한 한 끼에 담긴 정성이",
    "기분 좋게 즐긴 음식의 여운이",
    "든든히 채운 한 끼의 에너지가",
    "눈과 입이 즐거운 이 순간이",
    "오늘 고른 맛있는 메뉴가",
    "한 입 한 입 느낀 즐거움이",
    "정답게 차려진 음식의 온기가",
    "잠시 쉬어 가며 즐긴 한 끼가",
    "산뜻하게 시작한 식사 시간이",
    "행복을 담은 오늘의 음식이",
)

MESSAGE_ENDINGS = (
    "남은 하루도 환하게 밝혀주길 바라요.",
    "기분 좋은 일로 이어지길 바라요.",
    "오늘을 더 특별하게 만들어줄 거예요.",
    "마음까지 포근하게 채워주길 바라요.",
    "새로운 힘과 웃음을 선물해줄 거예요.",
    "여유롭고 행복한 하루를 만들어줄 거예요.",
    "하는 일마다 좋은 흐름으로 이어지길 바라요.",
    "몸과 마음에 건강한 활력을 더해줄 거예요.",
    "작지만 확실한 행복으로 오래 남길 바라요.",
    "오늘의 멋진 선택으로 기억되길 바라요.",
    "편안하고 즐거운 시간을 데려오길 바라요.",
    "당신의 하루에 달콤한 미소를 더해줄 거예요.",
    "힘찬 하루를 이어갈 든든한 응원이 될 거예요.",
    "좋은 사람들과 나눌 웃음으로 번지길 바라요.",
    "오늘 계획한 일에 산뜻한 힘을 보태줄 거예요.",
    "바쁜 하루 속 작은 쉼표가 되어주길 바라요.",
    "기억하고 싶은 맛있는 순간으로 남길 바라요.",
    "스스로를 아끼는 따뜻한 시간이 되길 바라요.",
    "내일을 기대하게 하는 기분 좋은 힘이 될 거예요.",
    "오늘도 잘 해낼 수 있다는 용기를 더해줄 거예요.",
)


st.set_page_config(
    page_title="뽀삐야 밥먹자 · 음식 이미지 분석",
    page_icon="🍽️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { --ink: #163f32; --muted: #687970; --paper: #fcfaf3; --mint: #e8f3e7; --line: #cfddd0; }
    .stApp { background: #fcfaf3; color: var(--ink); }
    .block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 2rem; }
    header[data-testid="stHeader"] { background: transparent; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg,#edf6eb 0%,#f8f7ed 100%); border-right: 1px solid #d6e1d4; }
    section[data-testid="stSidebar"] > div { padding: 2rem 1.25rem 1rem; }
    .hero {
        min-height: 210px; padding: 2rem 3rem; border: 1px solid #d4e2d2; border-radius: 24px;
        margin-bottom: 2rem; background: linear-gradient(110deg,#e5f1e4 0%,#f1f7ed 72%,#e8f3e7 100%);
        box-shadow: 0 10px 30px rgba(55,87,65,.045); display: flex; align-items: center;
        justify-content: center; gap: 2.2rem; position: relative; overflow: hidden;
    }
    .hero::before { content: "🌿"; position: absolute; left: 1.2rem; bottom: 1.2rem; font-size: 5rem; opacity: .11; }
    .hero::after { content: "♥"; position: absolute; right: 2.2rem; top: 2rem; color: #f69aa4; font-size: 2rem; }
    .hero-copy { max-width: 660px; text-align: center; }
    .hero h1 { color: #174938; margin: 0 0 .8rem; font-size: clamp(2.6rem,6vw,4rem); line-height: 1; letter-spacing: -.055em; }
    .hero p { color: #436557; margin: 0; font-size: 1rem; line-height: 1.75; }
    .character-frame {
        width: 180px; height: 180px; flex: 0 0 180px;
        display: flex; align-items: center; justify-content: center;
    }
    .character-image {
        display: block; width: 100%; height: 100%; object-fit: contain;
        filter: drop-shadow(0 12px 14px rgba(79, 86, 76, .12));
    }
    .section-kicker { color: #17533e; font-size: 1.08rem; font-weight: 850; margin: 1rem 0 .65rem; }
    .summary-card {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: .7rem;
        padding: 1rem; border: 1px solid #d4e1d2; border-radius: 18px;
        background: rgba(234,244,232,.72); margin: 0 0 .75rem;
    }
    .summary-item {
        padding: 1rem .8rem; text-align: center; border-right: 1px solid #d5ded3;
        background: #fffdf7;
    }
    .summary-item:first-child { border-radius: 13px 0 0 13px; }
    .summary-item:last-child { border-right: 0; border-radius: 0 13px 13px 0; }
    .summary-label { color: #63766c; font-size: .75rem; font-weight: 750; margin-bottom: .45rem; }
    .summary-value,.summary-calorie { color: #174c39; font-size: 1.35rem; font-weight: 900; }
    .calorie-note { color: #596c61; font-size: .78rem; line-height: 1.55; padding: .2rem .4rem; }
    .tip-note { padding: .75rem 1rem; border-radius: 10px; background: #e6f2e3; color: #486656; font-size: .8rem; }
    .calorie-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: .8rem; margin: .6rem 0 .45rem; }
    .calorie-card { padding: 1.25rem .9rem; border: 1px solid #d5dfd3; border-radius: 16px; background: #fffdf7; text-align: center; }
    .calorie-card:nth-child(2),.calorie-card:nth-child(3) { border-color: #ead8ad; }
    .calorie-card:nth-child(4) { border-color: #efced1; background:#fff8f7; }
    .calorie-card .icon { font-size: 2rem; }
    .calorie-card strong { display:block; color:#163f32; font-size:1rem; margin:.2rem 0; }
    .calorie-card span { color:#163f32; font-weight:850; }
    .small-note { color:#67766e; font-size:.75rem; margin-bottom:1rem; }
    .sidebar-title { color:#203d32; font-size:.78rem; margin-bottom:1.3rem; }
    .sidebar-heading { color:#153e31; font-size:1.35rem; font-weight:900; margin-bottom:1.25rem; }
    .side-row { display:grid; grid-template-columns:28px 1fr; gap:.55rem; margin:1.05rem 0; color:#24483a; }
    .side-row .side-icon { font-size:1.25rem; }
    .side-row strong { display:block; font-size:.84rem; }
    .side-row span { display:block; font-size:.78rem; margin-top:.15rem; }
    .side-note { padding:.9rem; margin:1.2rem 0; border-radius:12px; background:#e7f1e3; color:#4d6558; font-size:.72rem; line-height:1.75; }
    .side-character { display:block; width:145px; margin:2.2rem auto .4rem; filter:drop-shadow(0 8px 10px rgba(80,80,70,.1)); }
    .side-tagline { color:#ef8e95; font-size:.76rem; font-weight:800; text-align:center; }
    .daily-message { margin:1.35rem 0 .5rem; padding:1.25rem 1.4rem; border:1px solid #d4e2d2; border-radius:16px; background:linear-gradient(110deg,#edf6e9,#fffaf2); text-align:center; box-shadow:0 7px 20px rgba(55,87,65,.04); }
    .daily-message small { display:block; color:#668474; font-weight:800; margin-bottom:.35rem; }
    .daily-message strong { color:#244c3c; font-size:1rem; }
    .footer { text-align:center; color:#829087; font-size:.72rem; margin-top:1.5rem; }
    div[data-testid="stImage"] { width:100%; }
    div[data-testid="stImage"] img {
        width:100% !important; height:300px !important; object-fit:cover; object-position:center;
        border-radius:16px; box-shadow:0 8px 22px rgba(61,94,70,.06);
    }
    div[data-testid="stFileUploader"] { background:#fffdf8; border:1.5px dashed #a8c5b0; border-radius:16px; padding:.9rem 1rem; }
    div[data-testid="stProgress"] > div > div { background-color: #79a886; }
    @media (max-width: 700px) {
        .hero { padding:1.6rem; flex-direction:column; }
        .character-frame { width:150px; height:150px; flex-basis:150px; }
        .summary-card { grid-template-columns: 1fr; }
        .summary-item { border-right:0; border-bottom:1px solid #d5ded3; border-radius:10px !important; }
        .calorie-grid { grid-template-columns:repeat(2,1fr); }
        div[data-testid="stImage"] img { height:260px !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="분류 모델을 불러오는 중입니다…")
def get_classifier():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, class_names = load_model(MODEL_PATH, device)
    return model, transform, class_names, device


@st.cache_data
def get_metrics() -> dict:
    if not METRICS_PATH.is_file():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def get_confidence_threshold() -> float:
    if not CONFIDENCE_PATH.is_file():
        return 0.8
    return float(json.loads(CONFIDENCE_PATH.read_text(encoding="utf-8"))["threshold"])


@st.cache_data(show_spinner=False)
def classify_image_bytes(image_bytes: bytes) -> list[tuple[str, float]]:
    model, transform, class_names, device = get_classifier()
    with Image.open(BytesIO(image_bytes)) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].cpu().tolist()
    return sorted(zip(class_names, probabilities), key=lambda item: item[1], reverse=True)


def format_category_calories(category: str) -> str:
    return f"약 {CALORIES_BY_CATEGORY[category]} kcal"


def get_todays_message(image_bytes: bytes) -> str:
    """Return a stable, session-unique message for each uploaded image."""
    image_key = hashlib.sha256(image_bytes).hexdigest()
    assignments = st.session_state.setdefault("daily_message_assignments", {})
    if image_key not in assignments:
        message_count = len(MESSAGE_OPENINGS) * len(MESSAGE_ENDINGS)
        start_index = int(image_key, 16) % message_count
        used_indexes = set(assignments.values())
        message_index = start_index
        for offset in range(message_count):
            candidate = (start_index + offset) % message_count
            if candidate not in used_indexes:
                message_index = candidate
                break
        assignments[image_key] = message_index

    message_index = assignments[image_key]
    opening_index, ending_index = divmod(message_index, len(MESSAGE_ENDINGS))
    return f"{MESSAGE_OPENINGS[opening_index]} {MESSAGE_ENDINGS[ending_index]}"


@st.cache_data(show_spinner=False)
def get_character_data(path: str, modified_time: float) -> str:
    del modified_time
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


if not CHARACTER_PATH.is_file():
    st.error(f"캐릭터 이미지를 찾을 수 없습니다: {CHARACTER_PATH}")
    st.stop()
character_data = get_character_data(str(CHARACTER_PATH), CHARACTER_PATH.stat().st_mtime)

st.markdown(
    f"""
    <section class="hero">
      <div class="hero-copy">
        <h1>뽀삐야 밥먹자</h1>
        <p>한 장의 음식 사진을 읽고, 밥 · 국 · 반찬 · 후식 중 가장 가까운 한 가지와 그 판단의 확률을 숫자로 보여드립니다.</p>
      </div>
      <div class="character-frame">
        <img class="character-image" src="data:image/png;base64,{character_data}" alt="숟가락과 포크를 든 캐릭터">
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

metrics = get_metrics()
with st.sidebar:
    model_name = "EfficientNet-B0" if metrics and "efficientnet_b0" in metrics.get("model", "") else "ResNet-18"
    test_accuracy = metrics.get("test_accuracy", 0) if metrics else 0
    source_rows = metrics.get("source_rows", 0) if metrics else 0
    st.markdown(
        f"""
        <div class="sidebar-title">뽀삐야 밥먹자</div>
        <div class="sidebar-heading">분석 기준</div>
        <div class="side-row"><div class="side-icon">♨</div><div><strong>분류 항목</strong><span>밥 · 국 · 반찬 · 후식</span></div></div>
        <div class="side-row"><div class="side-icon">⚙</div><div><strong>분석 모델</strong><span>{model_name}</span></div></div>
        <div class="side-row"><div class="side-icon">◇</div><div><strong>테스트 정확도</strong><span>{test_accuracy:.1%}</span></div></div>
        <div class="side-row"><div class="side-icon">▣</div><div><strong>학습 이미지</strong><span>{source_rows:,}장</span></div></div>
        <hr>
        <div class="side-note">현재 숫자는 칼로리가 아니라 이미지 분류 확률입니다.<br>촬영 환경과 음식 구성에 따라 결과가 달라질 수 있습니다.</div>
        <img class="side-character" src="data:image/png;base64,{character_data}" alt="숟가락과 포크를 든 캐릭터">
        <div class="side-tagline">“오늘도 맛있는 한 끼 하세요!”</div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-kicker">사진 올리기</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "음식 사진을 클릭해서 선택하거나 이곳으로 드래그해 주세요",
    type=["jpg", "jpeg", "png", "webp"],
    help="사진을 올리면 별도의 선택 과정 없이 카테고리와 예상 칼로리를 바로 보여줍니다.",
)

if uploaded_file is None:
    st.caption("사진을 올리면 업로드한 이미지와 분석 결과가 여기에 표시됩니다.")
else:
    try:
        image_bytes = uploaded_file.getvalue()
        with Image.open(BytesIO(image_bytes)) as source_image:
            source_image.load()
            image = source_image.copy()
        with st.spinner("음식을 살펴보는 중입니다…"):
            scores = classify_image_bytes(image_bytes)
        prediction, confidence = scores[0]
        confidence_threshold = get_confidence_threshold()
        calorie_text = format_category_calories(prediction)
        image_column, result_column = st.columns([1, 1.35], gap="large")
        with image_column:
            st.markdown('<div class="section-kicker">업로드한 사진</div>', unsafe_allow_html=True)
            st.image(image, caption=uploaded_file.name, width="stretch")
        with result_column:
            st.markdown('<div class="section-kicker">분석 결과</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="summary-card">
                  <div class="summary-item"><div class="summary-label">예측 카테고리</div><div class="summary-value">{CATEGORY_ICON[prediction]} {prediction}</div></div>
                  <div class="summary-item"><div class="summary-label">AI 확신도</div><div class="summary-value">{confidence:.1%}</div></div>
                  <div class="summary-item"><div class="summary-label">예상 칼로리 (평균)</div><div class="summary-calorie">{calorie_text}</div></div>
                </div>
                <p class="calorie-note">※ 칼로리는 해당 카테고리의 평균값을 기준으로 제공합니다.</p>
                <div class="tip-note">💡 음식의 양, 종류, 조리 방법에 따라 실제 칼로리는 달라질 수 있어요.</div>
                """,
                unsafe_allow_html=True,
            )
            if confidence < confidence_threshold:
                st.warning(f"확신도가 {confidence_threshold:.0%}보다 낮아 결과 확인이 필요합니다.")

        st.markdown('<div class="section-kicker">카테고리별 평균 칼로리</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="calorie-grid">
              <div class="calorie-card"><div class="icon">🍚</div><strong>밥</strong><span>약 300 kcal</span></div>
              <div class="calorie-card"><div class="icon">🍲</div><strong>국</strong><span>약 120 kcal</span></div>
              <div class="calorie-card"><div class="icon">🍱</div><strong>반찬</strong><span>약 180 kcal</span></div>
              <div class="calorie-card"><div class="icon">🍎</div><strong>후식</strong><span>약 100 kcal</span></div>
            </div>
            <div class="small-note">※ 위 칼로리는 일반적인 평균값이며, 음식의 종류와 조리 방법에 따라 차이가 있습니다.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-kicker">네 가지 확률</div>', unsafe_allow_html=True)
        probability_columns = st.columns(4, gap="small")
        for index, (category, probability) in enumerate(scores):
            with probability_columns[index]:
                with st.container(border=True):
                    st.write(f"**{category}**")
                    st.progress(probability)
                    st.caption(f"{probability:.1%}")
        st.markdown(
            f"""
            <div class="daily-message">
              <small>오늘의 한마디</small>
              <strong>{get_todays_message(image_bytes)}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="footer">© 2026 뽀삐야 밥먹자. All rights reserved.</div>', unsafe_allow_html=True)
    except (UnidentifiedImageError, OSError):
        st.error("이미지를 읽을 수 없습니다. 정상적인 JPG, PNG 또는 WEBP 파일인지 확인해 주세요.")
    except Exception as error:
        st.error(f"예측 중 오류가 발생했습니다: {error}")

