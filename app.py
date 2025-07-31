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

# 🔥 변경: 분리된 모듈 임포트
from rag.loader import load_and_split
from rag.embedder import get_embeddings, create_vectorstore
from rag.retriever import HybridRetriever
from rag.translator import build_translator_chain

# 1) 환경 변수 로드
load_dotenv()

# 2) LangChain Tracer 생성
tracer = LangChainTracer()

# 캐시 디렉토리 생성
os.makedirs(".cache/files", exist_ok=True)
os.makedirs(".cache/embeddings", exist_ok=True)

st.title("PDF 기반 QA💬")

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = None

# 사이드바 UI
with st.sidebar:
    clear_btn      = st.button("대화 초기화")
    uploaded_files = st.file_uploader(
        "PDF 파일 업로드", type=["pdf"], accept_multiple_files=True
    )
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    )
    temperature = st.slider(
        "온도 설정 (temperature)", 0.0, 1.0, 0.0, 0.01
    )
    # 번역 체인 생성
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
    # 🔥 변경: retriever_fn 전달
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

# 메인 로직
if uploaded_files:
    # 🔥 변경: 로더·임베더·리트리버 모듈 사용
    split_docs = load_and_split(uploaded_files)
    embeddings = get_embeddings()
    vectorstore = create_vectorstore(split_docs, embeddings)
    vec_retriever = vectorstore.as_retriever()
    hybrid = HybridRetriever(vec_retriever, split_docs)

    # RAG 체인 생성 (메서드 전달)
    st.session_state["rag_chain"] = create_chain(
        hybrid.get_relevant_documents,
        model_name=selected_model,
        temperature=temperature
    )

if clear_btn:
    st.session_state["messages"].clear()

print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요!")
warning = st.empty()

if user_input:
    # 1) 번역
    eng_q = translator_chain.predict(text=user_input)
    st.chat_message("assistant").write(f"🔄 번역된 질문: {eng_q}")

    # 2) RAG 실행
    rag_chain = st.session_state["rag_chain"]
    if rag_chain:
        st.chat_message("user").write(user_input)
        response = rag_chain.stream(eng_q)
        with st.chat_message("assistant"):
            container, ans = st.empty(), ""
            for token in response:
                ans += token
                container.markdown(ans)
        add_message("user", user_input)
        add_message("assistant", ans)
    else:
        warning.error("PDF 파일을 먼저 업로드 해주세요.")
