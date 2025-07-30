import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.tracers import LangChainTracer
from translator import build_translator_chain

from dotenv import load_dotenv
import os
import yaml

# 환경 변수 로드
load_dotenv()

# 2) LangChainTracer 생성 (환경 변수에서 PROJECT_NAME, RUN_NAME 읽음)
tracer = LangChainTracer()

# 캐시 디렉토리 생성
os.makedirs(".cache/files", exist_ok=True)
os.makedirs(".cache/embeddings", exist_ok=True)

st.title("PDF 기반 QA💬")

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 사이드바 UI
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
    temperature = st.slider(
        "온도 설정 (temperature)",
        min_value=0.0, max_value=1.0,
        value=0.0, step=0.01
    )
    # 번역 체인 생성
    translator_chain = build_translator_chain(
        model_name=selected_model,
        temperature=temperature
    )

def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)

def add_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))

@st.cache_resource(show_spinner="문서 로드 및 분할 중입니다...")
def load_and_split(files):
    all_docs = []
    for file in files:
        path = f".cache/files/{file.name}"
        with open(path, "wb") as f:
            f.write(file.read())
        docs = PDFPlumberLoader(path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        all_docs.extend(splitter.split_documents(docs))
    return all_docs

@st.cache_resource(show_spinner="임베딩 객체 생성 중입니다...")
def get_embeddings():
    return OpenAIEmbeddings()

@st.cache_resource(show_spinner="벡터스토어(FAISS) 생성 중입니다...")
def create_vectorstore(_documents, _embeddings):
    return FAISS.from_documents(_documents, _embeddings)

def create_chain(retriever, model_name="gpt-4o", temperature=0.0):
    # YAML에서 프롬프트 스펙 로드
    with open("prompts/pdf-rag.yaml", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    prompt = PromptTemplate(
        template=spec["template"],
        input_variables=spec["input_variables"],
    )
    # callback_manager로 tracer 지정 → 모든 LLM 호출이 LangSmith에 기록됩니다
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        callbacks=[tracer]
        )
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# 메인 로직
if uploaded_files:
    # 1) 문서 로드 및 분할
    split_docs = load_and_split(uploaded_files)
    # 2) 임베딩 객체 생성
    embeddings = get_embeddings()
    # 3) 벡터스토어 생성
    vectorstore = create_vectorstore(split_docs, embeddings)
    retriever = vectorstore.as_retriever()
    # 4) 번역 체인 + 기존 RAG 체인 합성
    rag_runnable = create_chain(
        retriever,
        model_name=selected_model,
        temperature=temperature
    )
    # 5) 체인 초기화
    st.session_state["chain"] = create_chain(
        retriever,
        model_name=selected_model,
        temperature=temperature
    )

if clear_btn:
    st.session_state["messages"].clear()

print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요!")
warning = st.empty()

if user_input:
    chain = st.session_state["chain"]
    if chain is not None:
        st.chat_message("user").write(user_input)
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            container = st.empty()
            answer = ""
            for token in response:
                answer += token
                container.markdown(answer)
        add_message("user", user_input)
        add_message("assistant", answer)
    else:
        warning.error("PDF 파일을 먼저 업로드 해주세요.")
