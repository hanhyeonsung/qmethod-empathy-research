import streamlit as st
import pandas as pd
import datetime
import os, base64,json
import plotly.graph_objects as go
import numpy as np 
import re 
import time 
import requests 

# -----------------------------
# Page & Globals
# -----------------------------
st.set_page_config(page_title="Q-Method (SNS Research) Analyzer", layout="wide")
st.title("Q-Method (SNS Research) Analyzer")

DATA_PATH = "survey_data.csv"   # 로컬 CSV 경로
MIN_N_FOR_ANALYSIS = 5
TOPK_STATEMENTS = 5
EPS = 1e-8

# -----------------------------
# 기본 설정
# -----------------------------
DATA_PATH = "survey_data.csv"
Q_SET = [
    # 유사성 기반
    "내가 본 콘텐츠와 유사한 항목을 추천해주는 것은 공정하다고 느껴진다.",
    "이전에 내가 ‘좋아요’를 누른 콘텐츠와 비슷한 추천은 신뢰할 수 있다.",
    "팔로우한 계정과 관련된 콘텐츠를 추천받는 건 자연스럽다.",
    "내 취향과 유사한 콘텐츠가 추천되면 만족감이 높다.",

    # 사회적 증거 기반
    "다른 사람들이 많이 본 콘텐츠를 추천받으면 신뢰가 간다.",
    "내 친구들이 본 콘텐츠를 추천해주는 것은 나에게도 유용하다.",
    "많은 사용자들이 선호한 콘텐츠는 나도 선호할 가능성이 높다.",
    "인기 있는 콘텐츠를 추천해주는 건 설득력이 있다.",

    # 메커니즘/알고리즘 기반
    "추천 알고리즘이 내 행동 데이터를 분석해 콘텐츠를 제안하는 건 설득력 있다.",
    "AI가 내 반응 패턴을 기반으로 추천했다면 더 신뢰가 간다.",
    "추천 시스템이 추천 이유를 구체적으로 설명하면 신뢰가 생긴다.",
    "내 행동 데이터를 기반으로 작동하는 추천은 더 정교하다고 느껴진다.",

    # 개인화/맥락 기반
    "내가 자주 반응하는 시간대의 콘텐츠를 추천받는 건 유용하다.",
    "나의 위치나 기기 사용 맥락을 반영한 추천은 공정하게 느껴진다.",
    "내가 최근에 관심을 가진 주제를 기반으로 한 추천은 더 의미 있다.",
    "내 피드의 맥락에 맞는 콘텐츠가 추천되면 신뢰할 수 있다.",

    # 정당성/공정성 기반
    "추천이 어떻게 결정됐는지 설명이 없으면 불안하다.",
    "추천 기준이 공개되면 시스템을 더 신뢰할 수 있다.",
    "누구에게나 동일한 기준이 적용된 추천이 더 공정하다.",
    "내가 원하면 추천 기준을 확인할 수 있어야 한다고 생각한다.",

    # 투명성 및 윤리 기준
    "AI 추천이 광고나 상업적 목적 없이 운영되었으면 좋겠다.",
    "알고리즘이 편향되지 않았다는 설명이 있으면 더 신뢰할 수 있다.",
    "추천 알고리즘은 차별 없이 콘텐츠를 보여줘야 한다.",
    "AI가 추천 기준에 민감한 정보를 사용하지 않는 것이 중요하다.",

    # 직관/감정 기반
    "이유는 몰라도 내 취향에 맞으면 추천은 괜찮다고 생각한다.",
    "복잡한 설명보다는 ‘이 콘텐츠가 당신에게 맞아요’라는 식의 설명이 더 좋다.",
    "추천이 내 직관과 맞지 않으면 설명이 있어도 신뢰하기 어렵다.",
    "설명이 너무 기술적이면 오히려 이해하기 어렵다."
]
Q_COLS = [f"Q{i:02d}" for i in range(1, len(Q_SET)+1)]
LIKERT = ["전혀 동의하지 않음(1)", "동의하지 않음(2)", "보통(3)", "동의함(4)", "매우 동의함(5)"]
LIKERT_MAP = {k: i+1 for i, k in enumerate(LIKERT)}
MAX_COUNT = {1: 4, 2: 6, 3: 8, 4: 6, 5: 4}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")



# -----------------------------
# Secrets (GitHub)
# -----------------------------
def _get_secret(path, default=""):
    try:
        cur = st.secrets
        for key in path.split("."):
            cur = cur[key]
        return cur
    except Exception:
        return default

GH_TOKEN   = _get_secret("github.token")
GH_REPO    = _get_secret("github.repo")
GH_BRANCH  = _get_secret("github.branch", "main")
GH_REMOTEP = _get_secret("github.data_path", "survey_data.csv")  # 원격 저장 경로
GH_README  = _get_secret("github.readme_path", "README.md")         # (옵션)

# -----------------------------
# 세션 상태 초기화
# -----------------------------
if 'answers' not in st.session_state:
    st.session_state['answers'] = {f"Q{i:02d}": 3 for i in range(1, len(Q_SET)+1)}
if 'fid' not in st.session_state:
    st.session_state['fid'] = "TEST_FID"
if 'chosen_quarter' not in st.session_state:
    st.session_state['chosen_quarter'] = None
if 'email' not in st.session_state:
    st.session_state['email'] = ""
if 'passed_screen' not in st.session_state:
    st.session_state['passed_screen'] = False  # 기본값: False
if 'current_tab' not in st.session_state:
    st.session_state['current_tab'] = 0  # 기본 Tab0

# -----------------------------
# 실시간 척도 현황 계산
# -----------------------------
def calc_scale_counts(answers):
    counts = {i: 0 for i in range(1, 6)}
    for v in answers.values():
        counts[v] += 1
    return counts



def _gh_headers(token):
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "streamlit-qmethod-sns"
    }

def gh_get_sha(owner_repo, path, token, branch):
    url = f"https://api.github.com/repos/{owner_repo}/contents/{path}"
    r = requests.get(url, headers=_gh_headers(token), params={"ref": branch}, timeout=20)
    if r.status_code == 200:
        try:
            return r.json().get("sha")
        except Exception:
            return None
    elif r.status_code == 404:
        return None
    else:
        raise RuntimeError(f"GitHub GET 실패: {r.status_code} {r.text}")

def gh_put_file(owner_repo, path, token, branch, content_bytes, message):
    url = f"https://api.github.com/repos/{owner_repo}/contents/{path}"
    b64 = base64.b64encode(content_bytes).decode("ascii")
    sha = gh_get_sha(owner_repo, path, token, branch)
    payload = {"message": message, "content": b64, "branch": branch}
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=_gh_headers(token), data=json.dumps(payload), timeout=30)
    if r.status_code in (200, 201):
        return True, r.json()
    return False, f"{r.status_code}: {r.text}"

def push_csv_to_github(local_path, remote_path=None, note="Update survey_data.csv"):
    if not (GH_TOKEN and GH_REPO):
        return False, "GitHub secrets 누락(github.token, github.repo)"
    if remote_path is None:
        remote_path = GH_REMOTEP
    try:
        with open(local_path, "rb") as f:
            content = f.read()
    except Exception as e:
        return False, f"로컬 CSV 읽기 실패: {e}"
    ok, resp = gh_put_file(GH_REPO, remote_path, GH_TOKEN, GH_BRANCH, content, note)
    return ok, resp

# -----------------------------
# 사이드바 : 실시간 현황 패널
# -----------------------------


with st.sidebar:
    st.subheader("🔐 관리자 / 동기화")
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    admin_pw = st.sidebar.text_input("관리자 비밀번호 (선택)", type="password")
    if st.sidebar.button("로그인"):
        if admin_pw and _get_secret("admin.password") == admin_pw:
            st.session_state.authenticated = True
            st.sidebar.success("인증 성공")
        else:
            st.sidebar.error("인증 실패")

    auto_sync = st.sidebar.checkbox("응답 저장 시 GitHub 자동 푸시", value=True)

    st.subheader("📊 실시간 척도 현황")
    counts = calc_scale_counts(st.session_state['answers'])
    if st.button("🔄 새로고침"):
        st.rerun()

    df_counts = pd.DataFrame({
        "척도": [LIKERT[i-1] for i in range(1, 6)],
        "선택 문항 수": [counts[i] for i in range(1, 6)],
        "최대 허용 개수": [MAX_COUNT[i] for i in range(1, 6)],
    })

    st.dataframe(df_counts, width="content")

    # Plotly 그래프 표시
    fig = go.Figure(data=[
        go.Bar(name="선택 문항 수", x=LIKERT, y=[counts[i] for i in range(1,6)], marker_color='skyblue'),
        go.Bar(name="최대 허용 개수", x=LIKERT, y=[MAX_COUNT[i] for i in range(1,6)], marker_color='salmon')
    ])
    fig.update_layout(
        barmode='group',
        yaxis_title="문항 수",
        xaxis_tickangle=-20,
        template="plotly_white",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Utils
# -----------------------------



def is_valid_email(s: str) -> bool:
    if not s: return False
    s = s.strip()
    if len(s) > 150: return False
    return bool(EMAIL_RE.match(s))

def load_csv_safe(path: str):
    if not os.path.exists(path):
        return None
    try:
        if os.path.getsize(path) == 0:
            return None
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def save_csv_safe(df: pd.DataFrame, path: str):
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return True
    except Exception as e:
        st.error(f"CSV 저장 실패: {e}")
        return False

def ensure_q_columns(df: pd.DataFrame, q_count: int):
    cols = [f"Q{i:02d}" for i in range(1, q_count + 1)]
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    return df, cols

def zscore_rows(a: np.ndarray):
    m = a.mean(axis=1, keepdims=True)
    s = a.std(axis=1, ddof=0, keepdims=True)
    s = np.where(s < EPS, 1.0, s)
    return (a - m) / s

def rank_rows(a: np.ndarray):
    df = pd.DataFrame(a)
    return df.rank(axis=1, method="average", na_option="keep").values

def varimax(Phi, gamma=1.0, q=100, tol=1e-6, seed=42):
    Phi = Phi.copy(); p, k = Phi.shape
    R = np.eye(k); d_old = 0
    for _ in range(q):
        Lambda = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma/p) * (Lambda @ np.diag(np.sum(Lambda**2, axis=0))))
        )
        R = u @ vh
        d = np.sum(s)
        if d_old != 0 and d/d_old < 1 + tol: break
        d_old = d
    return Phi @ R, R

def choose_n_factors(eigvals, nmax):
    k = int(np.sum(eigvals >= 1.0))
    return max(2, min(nmax, k))




# -----------------------------
# Tabs
# -----------------------------

tab_titles = ["🧾 응답자 사전 조사", "✍️ 설문 수집", "📊 사람 요인화(Q) 분석", "☁️ GitHub 동기화 로그"]

if st.session_state['passed_screen']:
    default_tab = 1  # 사전조사 통과 → Tab1로
else:
    default_tab = 0  # 기본 Tab0

tabs = st.tabs(tab_titles)
current_tab = st.session_state.get('current_tab', default_tab)


# -----------------------------
# Tab0: Screen out / Quarter Over
# -----------------------------

# 상태 3가지
# 각 상태별 코드는 다음과 같습니다.

# 설문완료 : comp
# 쿼터오버 : quotafull
# 스크린아웃 : scrout
# 설문완료 url 예시  https://datain.co.kr/panel/panel.html?fid=test&status=comp 
# 쿼터오버 url 예시  https://datain.co.kr/panel/panel.html?fid=test&status=quotafull 
# 스크린아웃 url 예시  https://datain.co.kr/panel/panel.html?fid=test&status=scrout 

# 1. 응답자가 속한 쿼터별로 쿼터오버 여부를 판단 기능 +  쿼터오버라면 메일상의 url(status=qutoafull)로 리다이렉션하는 기능 구현

# 기존 표본설계대로 4유형 3조건(A/B/C)로 응답자가 각 쿼터로 나눠지고 쿼터별 최대 응답자수(30~35명)를 넘어서면 status=qutoafull

# 2. 스크린아웃 설문 구현 / 아래 내용중 

# 동일 연구 참여 여부 > 있으면 스크린아웃
# 미성년자 여부 > 해당 > 스크린아웃
# 정신 질환 진료 여부 > 해당 >  스크린아웃
# 일 평균 sns 방문 빈도 10회 이상 / 이하 (타 연구 수치 참고.)> 미만이면 스크린아웃
# 추천서비스 체험/경험 여부> 없으면 스크린아웃


# -----------------------------
# Tab0: Screen out / Quarter Over  (대체할 코드)
# -----------------------------
with tabs[0]:
    st.subheader("🙋‍♂️📋 응답자 사전 조사")

    # --- 임시 fid 설정 (실연동 시 아래처럼 교체)
    # fid = st.query_params.get("fid", ["unknown"])[0]
    # fid = st.text_input("테스트용 fid (실연동 시 URL의 ?fid=값으로 대체)", value=st.session_state.get("fid", "TEST_FID"))

    fid = st.query_params.get("fid", ["unknown"])[0]
    st.session_state['fid'] = fid

    st.markdown("### ⚠️ 사전 체크 항목 (임시 인터페이스)")
    st.info("※ 현재는 임시 구현입니다. 실제 연동 시에는 fid를 URL에서 받아 처리하고, 리디렉션 후 패널사에서 응답 저장/후속처리를 수행합니다.")

    # --- 기본 응답자 정보 입력 (스크린아웃/쿼터 판단에 사용)
    email_check = st.text_input("이메일 (중복체크용, 반드시 입력)")
    age = st.number_input("나이 (숫자)", min_value=0, max_value=120, value=30)
    mental_illness = st.selectbox("정신질환 진료 이력 (있으면 스크린아웃)", ["없음", "있음"])
    sns_freq = st.number_input("일 평균 SNS 방문 횟수 (숫자, 예: 12)", min_value=0, max_value=1000, value=10)
    rec_experience = st.selectbox("추천 서비스 사용 경험", ["있음", "없음"])
    # 임시로 응답자를 어느 쿼터(A/B/C)에 배치할지 선택하게 함 (실제 분류 로직이 있다면 그걸 사용)
    chosen_quarter = st.selectbox("응답자 쿼터 (테스트용 선택)", ["A", "B", "C"])

    st.markdown("---")
    st.write("**스크린아웃 / 쿼터오버 테스트 실행 버튼을 눌러 판정합니다.**")

    # 버튼 클릭 시 판정 수행
    if st.button("사전판정 실행"):
        # 0) 기본 입력 검증
        if not email_check or not is_valid_email(email_check):
            st.error("유효한 이메일을 입력하세요.")
            st.stop()

        # 1) 로컬 CSV 불러오기 (존재하면 데이터프레임, 아니면 빈 DF)
        df_existing = load_csv_safe(DATA_PATH)
        if df_existing is None:
            df_existing = pd.DataFrame(columns=Q_COLS + ["email", "ts", "quarter"])

        # 정리: quarter 컬럼이 있으면 공백 제거
        if "quarter" in df_existing.columns:
            df_existing["quarter"] = df_existing["quarter"].astype(str).str.strip().str.upper()

        # 2) 스크린아웃 조건들
        # 2-1) 동일 연구 참여 여부 (이메일 중복이면 스크린아웃)
        if email_check.strip() != "":
            emails_lower = df_existing.get("email", pd.Series([], dtype=str)).fillna("").astype(str).str.strip().str.lower()
            if email_check.strip().lower() in emails_lower.values:
                st.warning("해당 이메일로 이미 참여 이력이 있습니다. 스크린아웃 처리합니다.")
                st.markdown(f'<meta http-equiv="refresh" content="0; url=https://datain.co.kr/panel/panel.html?fid={fid}&status=scrout">', unsafe_allow_html=True)
                st.stop()

        # 2-2) 미성년자(예: 19세 미만) -> 스크린아웃
        if age < 19:
            st.warning("미성년자(19세 미만)는 조사 대상이 아닙니다. 스크린아웃 처리합니다.")
            st.markdown(f'<meta http-equiv="refresh" content="0; url=https://datain.co.kr/panel/panel.html?fid={fid}&status=scrout">', unsafe_allow_html=True)
            st.stop()

        # 2-3) 정신질환 진료 이력 -> 스크린아웃
        if mental_illness == "있음":
            st.warning("정신질환 진료 이력이 있으므로 스크린아웃 처리합니다.")
            st.markdown(f'<meta http-equiv="refresh" content="0; url=https://datain.co.kr/panel/panel.html?fid={fid}&status=scrout">', unsafe_allow_html=True)
            st.stop()

        # 2-4) 일 평균 SNS 방문 빈도 기준(예: 10회 미만 -> 스크린아웃)
        if sns_freq < 10:
            st.warning("일 평균 SNS 방문 빈도가 기준(10회) 미만입니다. 스크린아웃 처리합니다.")
            st.markdown(f'<meta http-equiv="refresh" content="0; url=https://datain.co.kr/panel/panel.html?fid={fid}&status=scrout">', unsafe_allow_html=True)
            st.stop()

        # 2-5) 추천서비스 체험/경험 여부 (없으면 스크린아웃)
        if rec_experience == "없음":
            st.warning("추천서비스 경험이 없는 응답자는 대상에서 제외됩니다. 스크린아웃 처리합니다.")
            st.markdown(f'<meta http-equiv="refresh" content="0; url=https://datain.co.kr/panel/panel.html?fid={fid}&status=scrout">', unsafe_allow_html=True)
            st.stop()

        # 3) 쿼터오버 판단 (지정한 쿼터에서 기존 응답 수 체크)
        # 목표: 각 쿼터 별 최대 허용수 = 5 (임시)
        QUOTA_LIMIT = 5

        # 현재 같은 쿼터에 해당하는 응답 수 집계 (대소문자/공백 제거)
        if "quarter" in df_existing.columns:
            q_counts = df_existing["quarter"].fillna("").astype(str).str.strip().str.upper().value_counts().to_dict()
        else:
            q_counts = {}

        current_count = q_counts.get(chosen_quarter.upper(), 0)

        st.write(f"현재 **쿼터 {chosen_quarter}** 응답 수: **{current_count}** (허용 최대 {QUOTA_LIMIT}명)")

        if current_count >= QUOTA_LIMIT:
            st.warning(f"쿼터 {chosen_quarter}이(가) 이미 가득 찼습니다. 쿼터오버 처리합니다.")
            st.markdown(f'<meta http-equiv="refresh" content="0; url=https://datain.co.kr/panel/panel.html?fid={fid}&status=quotafull">', unsafe_allow_html=True)
            st.stop()

        # 4) 통과 시 (설문 참여 가능)
        st.success("✅ 사전조건 통과 — 설문 참여 가능합니다. (임시 상태)")
        st.info("이제 Tab1(설문 수집)으로 이동하여 응답을 제출하세요. 실제 연동 시에는 설문 제출 후 패널사로 상태(comp) 리디렉션이 필요합니다.")
        st.session_state['passed_screen'] = True
        st.session_state['current_tab'] = 1
# -----------------------------
# Tab1: Survey
# -----------------------------
with tabs[1]:
    # ── 조사 안내 블록 (Streamlit) ──────────────────────────────────────────────────
    st.markdown("""
    ### 📢 **조사 안내**
    
    **주관:** 한국공학대학교 · 성균관대학교
    **소요 시간:** 약 **5–7분** │ **응답 형식:** 5점 리커트(28문항)
    
    ---
    
    #### 🎯 **연구 취지**
    SNS 기반 콘텐츠 추천 시스템에서 **추천 설명방식**이 사용자에게 어떻게 인식되는지를 파악하고,  
    이를 통해 사용자의 **공정성 인식** 및 **참여 행동**(클릭, 댓글, 공유 등)에 어떤 영향을 주는지를 분석합니다.
    
    본 조사에서는 추천 설명에 대한 인식 유형(예: 유사성 선호, 기계 기반 신뢰, 직관 수용 등)을 분류하며,  
    해당 결과는 후속 실험 설계의 조건 구분 또는 조절 변수로 활용됩니다.
    
    #### 📝 **참여 안내**
    - 28개 짧은 진술문에 대해 *전혀 동의하지 않음(1) ~ 매우 동의함(5)* 중 하나를 선택해 주세요. 중립적이거나 잘 몰라서 판단을 유보하고 싶은 경우에는 중간(3)으로 주로 유지해주세요. 결과적으로는 중간값이 가장 많도록 선택해주세요. 
    - 응답은 **익명 분석**을 원칙으로 하며, **이메일은 중복 제거 및 사후 안내**(예: 결과 공지, 보상 고지)에만 사용합니다.
    
    """)
    
    with st.expander("🔒 법적·윤리 안내 (펼쳐 보기)", expanded=False):
        st.markdown("""
    - **통계 목적 사용 원칙**: 응답은 **통계작성 및 학술연구 목적**에 한하여 사용되며, 법령에서 정한 경우를 제외하고 **제3자에게 제공되지 않습니다**.  
    - **개인정보 최소수집·분리보관**: 수집 항목은 설문 응답과 이메일 주소(중복 식별·사후 안내용)입니다. 이메일은 **분리 보관**되며 분석 자료에는 포함되지 않습니다.  
    - **익명 처리**: 분석 단계에서는 개인을 식별할 수 없도록 **비식별화/익명 처리**합니다.  
    - **자발적 참여·철회**: 참여는 자발적이며, **언제든 중단 또는 철회**할 수 있습니다(불이익 없음).  
    - **보관·파기**: 연구윤리 지침 및 관련 법령을 준수하여, 연구 종료 후 **정해진 보관기간**이 경과하면 안전하게 **파기**합니다.  
    - **관련 법령 준수**: 본 조사는 **통계 관련 법령** 및 **개인정보 보호 관련 법령**을 준수합니다.
    """)
    
    st.markdown("""
    ---
    **🧑‍🤝‍🧑 공동 연구기관:** **한국공학대학교 · 성균관대학교**
    """)
    # ────────────────────────────────────────────────────────────────────────────────

    if not st.session_state['passed_screen']:
        st.info("사전조사 통과 후에만 설문 참여가 가능합니다.")
    else:
        # 각 질문을 라디오로 표시 (answers는 session_state 사용)
        for i, stmt in enumerate(Q_SET, 1):
            qid = f"Q{i:02d}"
            sel_idx = st.session_state['answers'].get(qid, 3) - 1
            sel = st.radio(
                f"{i}. {stmt}",
                LIKERT,
                index=sel_idx,
                key=qid,
                horizontal=True
            )
            st.session_state['answers'][qid] = LIKERT_MAP[sel]

        # 제출 버튼
        if st.button("제출"):
            # 제출 시 한 번만 counts 계산
            counts = calc_scale_counts(st.session_state['answers'])
            if any(counts[k] > MAX_COUNT[k] for k in counts):
                st.warning("⚠️ 리커트 척도 제한을 초과했습니다. 수정 후 다시 제출하세요.")
                # Plotly 그래프 (optional, 최소화)
                fig = go.Figure(data=[
                    go.Bar(name="실제 선택 수", x=LIKERT, y=[counts[i] for i in range(1,6)], marker_color='skyblue'),
                    go.Bar(name="최대 허용 수", x=LIKERT, y=[MAX_COUNT[i] for i in range(1,6)], marker_color='salmon')
                ])
                fig.update_layout(barmode='group', yaxis_title="문항 수", xaxis_tickangle=-20, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # CSV 저장
                row = st.session_state['answers'].copy()
                row.update({
                    'ts': datetime.datetime.now().isoformat(),
                    'email': email_check,
                    'quarter': chosen_quarter
                })
                df = load_csv_safe(DATA_PATH)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True) if df is not None else pd.DataFrame([row])
                save_csv_safe(df, DATA_PATH)
                st.success("✅ 응답이 저장되었습니다!")
                time.sleep(1)

        # 종료 및 리디렉션 버튼
        if st.button("➡️ 응답 전송 및 설문 종료"):
            st.session_state['answers'] = {f"Q{i:02d}": 3 for i in range(1, len(Q_SET)+1)}
            st.session_state['email'] = ""
            st.session_state['chosen_quarter'] = None
            fid = st.session_state.get("fid", "TEST_FID")
            st.markdown(f'<meta http-equiv="refresh" content="0; url=https://datain.co.kr/panel/panel.html?fid={fid}&status=comp">', unsafe_allow_html=True)
            st.stop()

# -----------------------------
# Person-Q Analysis Helpers
# -----------------------------
def person_q_analysis(df_q: pd.DataFrame,
                      corr_metric: str = "Pearson",
                      n_factors: int | None = None,
                      rotate: bool = True):
    M = df_q.values.astype(float)
    if corr_metric.lower().startswith("spear"):
        M_proc = rank_rows(M)
        M_proc = zscore_rows(M_proc)
    else:
        M_proc = zscore_rows(M)

    R = np.corrcoef(M_proc, rowvar=True)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]

    nmax = max(2, min(6, R.shape[0]-1))
    if n_factors is None or n_factors < 1:
        n_factors = choose_n_factors(eigvals, nmax)
    else:
        n_factors = max(2, min(nmax, int(n_factors)))

    L = eigvecs[:, :n_factors] * np.sqrt(np.maximum(eigvals[:n_factors], 0))
    L_rot = varimax(L)[0] if rotate else L

    arrays = []
    for k in range(n_factors):
        w = np.clip(L_rot[:, k], a_min=0.0, a_max=None)
        if w.sum() <= EPS: w = np.abs(L_rot[:, k])
        w = w / (w.sum() + EPS)
        arr_k = w @ M_proc
        arrays.append(arr_k)
    arrays = np.vstack(arrays)
    return L_rot, eigvals, R, arrays

def assign_types(loadings: np.ndarray, emails: list[str], thr: float = 0.40, sep: float = 0.10):
    
    # N, K = loadings.shape
    # max_idx = loadings.argmax(axis=1)
    # max_val = loadings.max(axis=1)
    # sorted_vals = np.sort(loadings, axis=1)[:, ::-1]
    # second = sorted_vals[:,1] if K >= 2 else np.zeros(N)

    abs_loadings = np.abs(loadings)
    N, K = abs_loadings.shape
    max_idx = abs_loadings.argmax(axis=1)
    max_val = abs_loadings.max(axis=1)
    sorted_vals = np.sort(abs_loadings, axis=1)[:, ::-1]
    second = sorted_vals[:, 1] if K >= 2 else np.zeros(len(abs_loadings))

    assigned = (max_val >= thr) & ((max_val - second) >= sep)
    rows = []
    for i in range(N):
        rows.append({
            "email": emails[i] if i < len(emails) else f"id_{i}",
            "Type": f"Type{max_idx[i]+1}",
            "MaxLoading": float(max_val[i]),
            "Second": float(second[i]),
            "Assigned": bool(assigned[i])
        })
    return pd.DataFrame(rows)

def top_bottom_statements(factor_arrays: np.ndarray, topk=5):
    K, P = factor_arrays.shape
    tb = []
    for k in range(K):
        z = factor_arrays[k]
        top_idx = np.argsort(z)[::-1][:topk]
        bot_idx = np.argsort(z)[:topk]
        tb.append((top_idx, bot_idx, z))
    return tb

# -----------------------------
# Tab2: Person-Q Analysis
# -----------------------------
with tabs[2]:
    st.subheader("사람 요인화(Q) 분석")
    df = load_csv_safe(DATA_PATH)
    if df is None:
        st.info("아직 수집된 응답이 없습니다. (빈 파일이면 먼저 설문 제출)")
    else:
        df, _ = ensure_q_columns(df, q_count=len(Q_SET))
        df_q = df[Q_COLS].copy()
        mask = df_q.notna().sum(axis=1) >= int(0.6*len(Q_COLS))
        df_q = df_q[mask]
        emails = df.loc[mask, "email"].fillna("").astype(str).tolist()

        st.write(f"유효 응답자 수: **{len(df_q)}명**")
        if len(df_q) < MIN_N_FOR_ANALYSIS:
            st.warning(f"분석에는 최소 {MIN_N_FOR_ANALYSIS}명의 응답이 필요합니다.")
        else:
            with st.expander("⚙️ 분석 옵션", expanded=True):
                colA, colB, colC = st.columns(3)
                with colA:
                    corr_metric = st.selectbox("상관계수", ["Pearson", "Spearman"], index=0)
                with colB:
                    n_f_override = st.number_input("요인 수(선택, 0=자동)", min_value=0, max_value=6, value=0, step=1)
                    n_factors = None if n_f_override == 0 else int(n_f_override)
                with colC:
                    rotate = st.checkbox("Varimax 회전", value=True)

                thr = st.slider("유형 배정 임계값(최대 적재치)", 0.20, 0.70, 0.40, 0.05)
                sep = st.slider("1등-2등 적재치 최소 격차", 0.00, 0.50, 0.10, 0.05)

            try:
                loadings, eigvals, R, arrays = person_q_analysis(df_q, corr_metric, n_factors, rotate)
                K = loadings.shape[1]

                st.markdown(f"**추출 요인 수: {K}**")
                load_df = pd.DataFrame(loadings, columns=[f"Type{i+1}" for i in range(K)])
                load_df.insert(0, "email", emails)
                st.dataframe(load_df.style.background_gradient(cmap="Blues", axis=None), width="stretch")

                assign_df = assign_types(loadings, emails, thr=thr, sep=sep)
                st.markdown("### 참가자 유형 배정")
                st.dataframe(assign_df, width="stretch")
                st.write("유형별 인원수:", assign_df[assign_df["Assigned"]].groupby("Type").size().to_dict())

                st.download_button(
                    "📥 참가자-유형 배정 CSV",
                    data=assign_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="person_type_assignments.csv",
                    mime="text/csv"
                )

                arrays_df = pd.DataFrame(arrays, columns=Q_COLS, index=[f"Type{i+1}" for i in range(K)])
                st.markdown("### 유형별 factor array (진술 z-프로파일)")
                st.dataframe(arrays_df, width="content")
                st.download_button(
                    "📥 유형별 factor array CSV",
                    data=arrays_df.to_csv().encode("utf-8-sig"),
                    file_name="type_factor_arrays.csv",
                    mime="text/csv"
                )

                st.markdown(f"### 유형별 상/하위 진술 Top {TOPK_STATEMENTS}")
                tb = top_bottom_statements(arrays, topk=TOPK_STATEMENTS)
                for i, (top_idx, bot_idx, z) in enumerate(tb, start=1):
                    with st.expander(f"Type{i} 상/하위 진술", expanded=True if i==1 else False):
                        st.markdown("**상위(+) 진술**")
                        for j in top_idx:
                            st.write(f"- Q{j+1:02d} (z={z[j]:.2f}) : {Q_SET[j]}")
                        st.markdown("**하위(−) 진술**")
                        for j in bot_idx:
                            st.write(f"- Q{j+1:02d} (z={z[j]:.2f}) : {Q_SET[j]}")

            except Exception as e:
                st.error(f"사람 요인화 분석 중 오류: {e}")

# -----------------------------
# Tab3: GitHub Sync Log / Manual Push
# -----------------------------
with tabs[3]:
    st.subheader("GitHub 동기화")
    if not (GH_TOKEN and GH_REPO):
        st.warning("Secrets에 github.token, github.repo 설정이 필요합니다.")
        st.code("""
[github]
token = "ghp_..."
repo  = "owner/repo"
branch = "main"
data_path = "survey_data.csv"
        """, language="toml")
    else:
        st.success(f"원격: {GH_REPO} @ {GH_BRANCH}\n경로: {GH_REMOTEP}")

    if st.button("지금 동기화(수동)"):
        if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0:
            ok, resp = push_csv_to_github(DATA_PATH, GH_REMOTEP,
                                          note=f"Manual sync {GH_REMOTEP} at {datetime.datetime.now().isoformat()}")
            if ok:
                st.success("GitHub에 동기화되었습니다.")
                st.json(resp)
            else:
                st.error(f"동기화 실패: {resp}")
        else:
            st.error("로컬 CSV가 없거나 비어있습니다. 먼저 설문을 제출해 CSV를 생성하세요.")
