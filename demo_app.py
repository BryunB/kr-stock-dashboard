"""데모 대시보드 진입점 — 모니터링(국내증시/코인) + 모의투자 화면을 하나로 묶는다.

PRD.md 5.6 참고. app.py는 지금 지인에게 공유 중인 배포이므로 한 글자도 수정하지 않는다.
"주가 스크리닝"은 app.py를 그대로 재사용하는 대신, 국내증시/코인을 함께 다루는
pages/모니터링.py(app.py의 상위 확장판)로 대체했다 — app.py 자체는 별도 배포로 여전히
독립 운영된다(이 파일이 참조하지 않을 뿐 삭제되거나 바뀌지 않았다). Streamlit Cloud에는
이 파일을 main file로 하는 별도 배포를 새로 추가한다(기존 app.py 배포는 그대로 유지).

실행: .venv\\Scripts\\streamlit.exe run demo_app.py
"""

import streamlit as st

st.set_page_config(page_title="주가 대시보드", layout="wide")

pg = st.navigation(
    [
        st.Page("pages/모니터링.py", title="모니터링", icon="🖥️"),
        st.Page("pages/모의투자.py", title="모의투자", icon="💰"),
    ]
)
pg.run()
