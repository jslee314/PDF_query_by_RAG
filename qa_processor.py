# qa_processor.py
import streamlit as st
from langchain_core.messages.chat import ChatMessage

def render(*, user_input: str, translator_chain, tracer):
    """
    질문 처리 및 답변 렌더링

    Keyword-only arguments:
    - user_input: 이전에 입력된 질문(한글)
    - translator_chain: 한글 질문을 영어로 번역하는 체인
    - tracer: LangChain 트레이서
    """
    st.header("2. 질문 입력 및 답변")

    # 문서 처리 모드가 완료되었는지 확인
    if not st.session_state.get("processed", False):
        st.warning("먼저 '문서 처리' 모드에서 파일을 업로드하고 처리하세요.")
        return

    # 질문 입력
    question = st.text_input("궁금한 내용을 입력하세요", value=user_input)
    if not question:
        return
    # 세션에 질문 저장
    st.session_state["last_question"] = question

    # 영어로 번역
    eng_q = translator_chain.predict(text=question)
    st.write(f"🔄 번역된 질문: {eng_q}")

    # 파일별 리트리버 상위 결과
    st.subheader("📂 파일별 유사도 상위 결과")
    for fname, retriever in st.session_state.get("file_retrievers", {}).items():
        with st.expander(fname):
            top_docs = retriever.get_relevant_documents(eng_q)[:3]
            for i, doc in enumerate(top_docs, start=1):
                source = doc.metadata.get("source", fname)
                page   = doc.metadata.get("page", "N/A")
                st.markdown(f"**결과 {i}** (출처: {source}, 페이지: {page})")
                st.write(doc.page_content[:300] + "…")

    # 파일별 RAG 답변 스트리밍
    st.subheader("📄 파일별 답변")
    for fname, chain in st.session_state.get("file_chains", {}).items():
        st.markdown(f"---\n**{fname}**")
        with st.chat_message("assistant"):
            container = st.empty()
            answer = ""
            for token in chain.stream(eng_q):
                answer += token
                container.markdown(answer)

    # 대화 기록에 추가
    st.session_state["messages"].append(ChatMessage(role="user", content=question))
    combined = "\n\n".join(
        f"## {fname}\n{chain.invoke(eng_q)}"
        for fname, chain in st.session_state.get("file_chains", {}).items()
    )
    st.session_state["messages"].append(ChatMessage(role="assistant", content=combined))

    # 이전 대화 출력
    for msg in st.session_state.get("messages", []):
        st.chat_message(msg.role).write(msg.content)
