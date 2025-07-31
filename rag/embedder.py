import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# @st.cache_resource(show_spinner="임베딩 객체 생성 중입니다...")
def get_embeddings():
    """OpenAIEmbeddings 인스턴스 반환"""
    return OpenAIEmbeddings()

# @st.cache_resource(show_spinner="벡터스토어(FAISS) 생성 중입니다...")
def create_vectorstore(_documents, _embeddings):
    """
    FAISS.from_documents 호출. 
    Streamlit 캐싱 오류 방지를 위해 매개변수 앞에 언더스코어 사용.
    """
    return FAISS.from_documents(_documents, _embeddings)