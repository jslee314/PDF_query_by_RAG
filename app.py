# app.py
import os
import yaml
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.tracers import LangChainTracer

from rag.loader import load_and_split
from rag.embedder import get_embeddings, create_vectorstore
from rag.retriever import HybridRetriever
from rag.translator import build_translator_chain

# 1) 환경 변수 로드
load_dotenv()

# 2) LangChain Tracer 생성
tracer = LangChainTracer()

# 캐시 디렉토리 준비
os.makedirs(".cache/files", exist_ok=True)
os.makedirs(".cache/embeddings", exist_ok=True)

st.title("PDF 기반 QA💬")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = None
# 🔥 변경: 업로드된 파일 트래킹용
if "uploaded_ids" not in st.session_state:
    st.session_state["uploaded_ids"] = None

# — 사이드바 UI —
with st.sidebar:
    clear_btn      = st.button("대화 초기화")
    uploaded_files = st.file_uploader(
        "PDF 파일 업로드", type=["pdf"], accept_multiple_files=True
    )
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    )
    temperature     = st.slider(
        "온도 설정 (temperature)", 0.0, 1.0, 0.0, 0.01
    )
    translator_chain = build_translator_chain(
        model_name=selected_model,
        temperature=temperature
    )

def print_messages():
    for m in st.session_state["messages"]:
        st.chat_message(m.role).write(m.content)

def add_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))

def create_chain(retriever_fn, model_name="gpt-4o", temperature=0.0):
    with open("prompts/pdf-rag.yaml", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    prompt = PromptTemplate(
        template=spec["template"],
        input_variables=spec["input_variables"],
    )
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        callbacks=[tracer]
    )
    return (
        {"context": retriever_fn, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# — 메인 로직 — 
# 🔥 변경: 초기화 블록 — 파일 업로드가 새로 감지될 때만 heavy 작업 실행
if uploaded_files:
    current_ids = tuple(f.name for f in uploaded_files)
    if st.session_state["uploaded_ids"] != current_ids:
        # 새 파일 업로드 감지 → 처리 시작
        st.session_state["uploaded_ids"] = current_ids
        # 문서별 프로그레스바
        total = len(uploaded_files)
        progress = st.progress(0)
        split_docs = []
        title_ph = st.empty()
        for idx, f in enumerate(uploaded_files):
            title_ph.info(f"🔄 처리 중: {f.name}")
            docs = load_and_split([f])  # cached
            split_docs.extend(docs)
            progress.progress(int((idx+1)/total * 100))
        title_ph.empty()
        progress.empty()

        # 임베딩
        with st.spinner("임베딩 모델 준비 중..."):
            embeddings = get_embeddings()
        # 벡터스토어
        with st.spinner("FAISS 스토어 생성 중..."):
            vectorstore = create_vectorstore(split_docs, embeddings)
        # 리트리버·체인 생성
        with st.spinner("RAG 체인 초기화 중..."):
            vec_ret = vectorstore.as_retriever()
            hybrid = HybridRetriever(vec_ret, split_docs)
            st.session_state["rag_chain"] = create_chain(
                hybrid.get_relevant_documents,
                model_name=selected_model,
                temperature=temperature
            )

if clear_btn:
    st.session_state["messages"].clear()
    # 🔥 변경: 업로드 리셋
    st.session_state["uploaded_ids"] = None
    st.session_state["rag_chain"]    = None

print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요!")
warning = st.empty()

if user_input:
    if st.session_state["rag_chain"] is None:
        warning.error("먼저 PDF를 업로드하고 처리하세요.")
    else:
        # 한글→영어 번역
        eng_q = translator_chain.predict(text=user_input)
        st.chat_message("assistant").write(f"🔄 번역 질문: {eng_q}")

        # RAG 호출
        st.chat_message("user").write(user_input)
        response = st.session_state["rag_chain"].stream(eng_q)
        with st.chat_message("assistant"):
            container, ans = st.empty(), ""
            for t in response:
                ans += t
                container.markdown(ans)
        add_message("user", user_input)
        add_message("assistant", ans)
