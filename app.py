import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate    
from dotenv import load_dotenv
import os
import yaml

# API KEY 정보 로드
load_dotenv()

# 캐시 디렉토리 생성
os.makedirs(".cache/files", exist_ok=True)
os.makedirs(".cache/embeddings", exist_ok=True)

st.title("PDF 기반 QA💬")

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 사이드바
with st.sidebar:
    clear_btn    = st.button("대화 초기화")
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0)

def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)

def add_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))

@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    path = f".cache/files/{file.name}"
    with open(path, "wb") as f:
        f.write(file.read())

    docs = PDFPlumberLoader(path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever()

def create_chain(retriever, model_name="gpt-4o"):
    # 1) YAML에서 prompt 스펙 로드
    with open("prompts/pdf-rag.yaml", encoding="utf-8") as f:
        prompt_spec = yaml.safe_load(f)

    # 2) PromptTemplate 객체 생성
    prompt_template = PromptTemplate(
        template=prompt_spec["template"],
        input_variables=prompt_spec["input_variables"],
    )

    # 3) LLM 객체
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 4) Runnable 파이프라인 구성
    #    retriever에서 context, RunnablePassthrough로 question 받아
    #    → PromptTemplate (Runnable) → LLM → StrOutputParser
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

# 파일 업로드 시
if uploaded_file:
    retriever = embed_file(uploaded_file)
    st.session_state["chain"] = create_chain(retriever, selected_model)

if clear_btn:
    st.session_state["messages"].clear()

print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요!")
warning = st.empty()

if user_input:
    chain = st.session_state["chain"]
    if chain:
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
        warning.error("파일을 업로드 해주세요.")
