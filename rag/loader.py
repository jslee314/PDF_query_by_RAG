
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter, SpacyTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader

@st.cache_resource(show_spinner="문서 로드 및 분할 중입니다...")
def load_and_split(_files,
                   loader_option: str = "PDFPlumberLoader",
                   splitter_option: str = "RecursiveCharacterTextSplitter",
                   cache_path: str = ".cache/files"):
    """
    여러 PDF 파일을 받아 로더 및 스플릿터 옵션에 따라 처리 후
    청크 리스트를 반환합니다.

    이 문서들(의료기기 가이드라인, NIST SP, FDA Cybersecurity 등)은 표와 레이아웃이 매우 중요하니, 다음과 같이 선택하는 걸 권장합니다:
    PDFPlumberLoader
        표(table)와 텍스트 블록, 레이아웃 정보를 잘 뽑아 줍니다.
        복잡한 규제 가이드라인의 표나 표제목, 섹션 구분을 그대로 살리기에 가장 무난한 선택입니다.

    UnstructuredPDFLoader
        텍스트·표·목록 등 요소별로 더 세밀하게 파싱하지만, 파싱 시간이 많이 걸립니다.
        “정말” 요소 단위로 구분이 필요하거나, 문서 내부 구조를 세밀하게 분석할 때 고려해 보세요.

    PyPDFLoader
        속도는 빠르지만, 모든 내용을 단순 텍스트로 뭉뚱그려 가져옵니다.
        표가 거의 없고 “순수 텍스트” 위주인 문서에만 권장합니다.

    결론:
    지금 올려주신 가이드라인·규제 문서는 표와 구조화된 섹션이 많으므로, 기본값으로 PDFPlumberLoader를 사용하시되, 필요 시 UnstructuredPDFLoader로 테스트해 보시는 걸 추천드립니다.


    PDF들은 모두 기술 가이드·규제 문서라서, 단순히 고정 글자 수로 자르기보다는 “문장ㆍ절” 단위나 “토큰 수” 단위로 의미 있는 경계를 살려 주는 스플릿터를 쓰는 게 좋습니다.
    SpacyTextSplitter
      문장 경계를 정확히 인식해 “한 문장씩” 잘라 줍니다. 
      단락이 길고 문장이 명확한 기술 문서에 잘 맞습니다.
    RecursiveCharacterTextSplitter (기본) 
        "\n\n" → "\n" → " " 순서로 가능한 긴 구분자 기준으로 최대한 문맥을 유지하며 자릅니다. 
        설정만 잘 조정해 주면, 헤더·목록·표 설명 등 레이아웃 단위도 그대로 보존됩니다.

    """
    all_docs = []
    for file in _files:
        path = f"{cache_path}/{file.name}"
        with open(path, "wb") as f:
            f.write(file.read())
        # 문서 로더 선택
        if loader_option == "PDFPlumberLoader":
            from langchain_community.document_loaders import PDFPlumberLoader
            docs = PDFPlumberLoader(path).load()
        elif loader_option == "PyPDFLoader":
            from langchain.document_loaders import PyPDFLoader
            loader = PyPDFLoader(path)
            docs = loader.load_and_split()
        elif loader_option == "UnstructuredPDFLoader":
            from langchain.document_loaders import UnstructuredPDFLoader
            loader = UnstructuredPDFLoader(path, mode="elements")
            docs = loader.load()
        else:
            raise ValueError(f"알 수 없는 로더 옵션: {loader_option}")
        # 텍스트 스플릿터 선택
        if splitter_option == "RecursiveCharacterTextSplitter":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " "]
            )
        elif splitter_option == "SpacyTextSplitter":
            from langchain_text_splitters import SpacyTextSplitter
            splitter = SpacyTextSplitter()
        elif splitter_option == "TokenTextSplitter":
            from langchain_text_splitters import TokenTextSplitter
            splitter = TokenTextSplitter(chunk_size=800, chunk_overlap=100)
        else:
            raise ValueError(f"알 수 없는 스플릿터 옵션: {splitter_option}")
        chunks = splitter.split_documents(docs)
        all_docs.extend(chunks)
    return all_docs

