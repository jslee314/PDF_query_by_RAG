# 프로젝트 구조

```
pdf_query/
├─ app.py
├─ index_processor.py
├─ qa_processor.py
├─ rag/
│  ├─ loader.py
│  ├─ embedder.py
│  ├─ retriever.py
│  └─ translator.py
├─ prompts/
│  └─ pdf-rag.yaml
├─ requirements.txt
├─ .env
└─ .vscode/
```

## 파일 설명

- **app.py**  
  - 애플리케이션 진입점
  - 사이드바에서 모드(문서 처리/질문 응답), 파일 업로드, 모델·온도·로더·스플릿터 옵션을 선택
  - 선택에 따라 `index_processor` 와 `qa_processor` 모듈을 호출하여 UI 흐름을 제어

- **index_processor.py**  
  - **문서 처리** 로직 담당
  - 업로드된 PDF마다 지정된 로더와 스플릿터로 텍스트 청크 생성
  - Embedding 생성, FAISS 벡터스토어 구성
  - RAG 체인 초기화 후 세션 상태에 저장

- **qa_processor.py**  
  - **질문 응답** 로직 담당
  - 한글 질문을 영어로 번역(Translator Chain)
  - 각 파일별로 유사도 상위 청크를 표시
  - RAG 체인을 이용해 답변을 스트리밍 형태로 출력
  - 대화 기록 관리 및 이전 메시지 렌더링

- **rag/loader.py**  
  - PDF 로더(`PDFPlumberLoader`, `PyPDFLoader`, `UnstructuredPDFLoader`)와
    스플릿터(`RecursiveCharacterTextSplitter`, `SpacyTextSplitter`, `TokenTextSplitter`) 옵션에 따라
    문서를 로드·분할하는 `load_and_split` 함수 제공

- **rag/embedder.py**  
  - OpenAI 임베딩 생성 함수(`get_embeddings`)
  - FAISS 벡터스토어 생성 함수(`create_vectorstore`)
  - Streamlit 캐시로 효율화

- **rag/retriever.py**  
  - 벡터 기반과 키워드 기반 리트리버를 결합한 `HybridRetriever` 클래스 정의
  - 중복 제거된 문서 청크를 반환

- **rag/translator.py**  
  - 한글 질문을 영어로 번역하는 Translator Chain(`build_translator_chain`)
  - `PromptTemplate`과 `ChatOpenAI`를 사용하여 번역 수행

- **prompts/pdf-rag.yaml**  
  - RAG 체인에서 사용하는 시스템·유저 프롬프트 템플릿 정의

- **requirements.txt**  
  - 프로젝트 의존성 목록

- **.env**  
  - OpenAI API 키 등 환경변수 설정 파일

- **.vscode/**  
  - VSCode 전용 설정
