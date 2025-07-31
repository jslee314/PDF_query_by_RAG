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

import index_processor
import qa_processor

# 1) 환경변수 로드
load_dotenv()

# 2) LangChain Tracer 생성
tracer = LangChainTracer()

# 캐시 디렉토리 준비
os.makedirs(".cache/files", exist_ok=True)
os.makedirs(".cache/embeddings", exist_ok=True)

st.title("PDF 기반 QA💬")

# 사이드바: 모드 선택
mode = st.sidebar.radio("모드 선택", ["문서 처리", "질문 응답"])

# 공통 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 사이드바: 파일 업로드, 모델, 온도, 로더/스플릿터 옵션 선택
uploaded_files = st.sidebar.file_uploader(
    "PDF 파일 업로드", type=["pdf"], accept_multiple_files=True
)
selected_model = st.sidebar.selectbox(
    "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
)
temperature = st.sidebar.slider(
    "온도 설정 (temperature)", 0.0, 1.0, 0.0, 0.01
)
loader_option = st.sidebar.selectbox(
    "문서 로더 선택",
    ["PyPDFLoader", "PDFPlumberLoader", "UnstructuredPDFLoader"],
    index=1
)
splitter_option = st.sidebar.selectbox(
    "스플릿터 선택",
    ["RecursiveCharacterTextSplitter", "SpacyTextSplitter", "TokenTextSplitter"],
    index=0
)

# 사이드바: YAML 프롬프트 로드 & 편집
with st.sidebar.expander("🔧 RAG 프롬프트 편집"):
    raw_spec = yaml.safe_load(open("prompts/pdf-rag.yaml", encoding="utf-8"))
    default_template = raw_spec["template"]
    input_vars       = raw_spec["input_variables"]
    edited_template  = st.text_area(
        "Prompt Template", value=default_template, height=200
    )

translator_chain = build_translator_chain(
    model_name=selected_model,
    temperature=temperature
)

# 모드별 렌더링
if mode == "문서 처리":
    index_processor.render(
        uploaded_files=uploaded_files,
        selected_model=selected_model,
        temperature=temperature,
        translator_chain=translator_chain,
        tracer=tracer,
        loader_option=loader_option,
        splitter_option=splitter_option,
        prompt_template=edited_template,
        input_variables=input_vars,
    )
elif mode == "질문 응답":
    qa_processor.render(
        user_input=st.session_state.get("last_question", ""),
        translator_chain=translator_chain,
        tracer=tracer
    )
