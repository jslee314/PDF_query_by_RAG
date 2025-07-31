import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# @st.cache_resource(show_spinner="문서 로드 및 분할 중입니다...")
def load_and_split(_files, cache_path=".cache/files"):
    """
    여러 PDF 파일을 받아, page 단위로 split_documents() 호출 후
    모든 청크 리스트로 반환.
    """
    all_docs = []
    for file in _files:
        path = f"{cache_path}/{file.name}"
        with open(path, "wb") as f:
            f.write(file.read())
        docs = PDFPlumberLoader(path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        all_docs.extend(splitter.split_documents(docs))
    return all_docs
