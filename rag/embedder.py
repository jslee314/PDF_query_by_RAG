from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def get_embeddings():
    """OpenAIEmbeddings 인스턴스 반환"""
    return OpenAIEmbeddings()

def create_vectorstore(documents, embeddings):
    """
    FAISS.from_documents 호출. 
    Streamlit 캐싱 오류 방지를 위해 매개변수 앞에 언더스코어 사용.
    """
    return FAISS.from_documents(documents, embeddings)