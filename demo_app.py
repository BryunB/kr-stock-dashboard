"""데모 대시보드 진입점 — 기존 app.py + 모의투자 화면을 하나로 묶는다.

PRD.md 5.6 참고. app.py는 지금 지인에게 공유 중인 배포이므로 한 글자도 수정하지
않는다 — st.Page("app.py")로 기존 스크립트를 그대로 페이지로 재사용한다. Streamlit
Cloud에는 이 파일을 main file로 하는 별도 배포를 새로 추가한다(기존 app.py 배포는
그대로 유지).

실행: .venv\\Scripts\\streamlit.exe run demo_app.py
"""

import streamlit as st

st.set_page_config(page_title="주가 대시보드", layout="wide")

pg = st.navigation(
    [
        st.Page("app.py", title="주가 스크리닝", icon="📊"),
        st.Page("pages/모의투자.py", title="모의투자", icon="💰"),
    ]
)
pg.run()
