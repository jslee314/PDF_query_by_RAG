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

# 1) 환경변수 로드
load_dotenv()

# 2) LangChain Tracer
tracer = LangChainTracer()

# 캐시 디렉토리 준비
os.makedirs(".cache/files", exist_ok=True)
os.makedirs(".cache/embeddings", exist_ok=True)

st.title("PDF 기반 QA💬")

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
# 🔥 변경: 파일별 분할/리트리버/체인 저장용
if "split_docs_by_file" not in st.session_state:
    st.session_state["split_docs_by_file"] = {}
if "file_retrievers" not in st.session_state:
    st.session_state["file_retrievers"] = {}
if "file_chains" not in st.session_state:
    st.session_state["file_chains"] = {}

with st.sidebar:
    clear_btn      = st.button("대화 초기화")
    uploaded_files = st.file_uploader(
        "PDF 파일 업로드",
        type=["pdf"],
        accept_multiple_files=True
    )
    selected_model = st.selectbox(
        "LLM 선택",
        ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"],
        index=0
    )
    temperature     = st.slider(
        "온도 설정 (temperature)",
        0.0, 1.0, 0.0, 0.01
    )
    translator_chain = build_translator_chain(
        model_name=selected_model,
        temperature=temperature
    )

def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)

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

# — 문서 처리: 업로드된 파일별로 분할, 임베딩, 리트리버, 체인 생성 — #
if uploaded_files:
    total = len(uploaded_files)
    progress = st.progress(0)
    split_docs_by_file = {}
    file_retrievers = {}
    file_chains = {}

    for idx, f in enumerate(uploaded_files):
        st.sidebar.info(f"⏳ 처리 중: **{f.name}**")
        # 1) 파일별 분할
        docs = load_and_split([f])  # 🔥 캐시됨
        split_docs_by_file[f.name] = docs

        # 2) 파일별 벡터스토어 & 리트리버
        vs = create_vectorstore(docs, get_embeddings())
        file_retrievers[f.name] = vs.as_retriever()

        # 3) 파일별 RAG 체인
        file_chains[f.name] = create_chain(
            file_retrievers[f.name],
            model_name=selected_model,
            temperature=temperature
        )
        progress.progress(int((idx + 1) / total * 100))

    progress.empty()
    st.session_state["split_docs_by_file"] = split_docs_by_file
    st.session_state["file_retrievers"]    = file_retrievers
    st.session_state["file_chains"]        = file_chains

if clear_btn:
    st.session_state["messages"].clear()
    st.session_state["split_docs_by_file"] = {}
    st.session_state["file_retrievers"]    = {}
    st.session_state["file_chains"]        = {}

print_messages()

# — 질문 처리 — #
user_input = st.chat_input("궁금한 내용을 물어보세요!")
warning = st.empty()

if user_input:
    if not st.session_state["file_chains"]:
        warning.error("먼저 PDF를 업로드하고 처리하세요.")
    else:
        # 1) 한글→영어 번역
        eng_q = translator_chain.predict(text=user_input)
        st.chat_message("assistant").write(f"🔄 번역된 질문: {eng_q}")

        # 2) 파일별 답변 스트리밍
        for fname, chain in st.session_state["file_chains"].items():
            st.markdown(f"### 📄 {fname}")
            with st.chat_message("assistant"):
                container = st.empty()
                answer = ""
                for token in chain.stream(eng_q):  # 🔥 stream() 사용
                    answer += token
                    container.markdown(answer)
            st.write("---")

        # 3) 대화 기록에 추가
        add_message("user", user_input)
        # 전체 파일별 답변을 하나로 합쳐 기록
        combined = "\n\n".join(
            f"## {fname}\n{chain.invoke(eng_q)}"
            for fname, chain in st.session_state["file_chains"].items()
        )
        add_message("assistant", combined)
