# PDF 기반 QA 시스템

이 프로젝트는 **LangChain**, **RAG**, **Streamlit**을 사용하여 PDF 문서를 업로드하고 자연어 질의 응답(QA)을 수행하는 웹 애플리케이션입니다.

## 주요 기능

- PDF 파일 업로드 및 문서 임베딩
- FAISS를 이용한 벡터 검색(Retriever)
- GPT-4o / GPT-4 Turbo 모델 기반 질의 응답
- 대화형 채팅 UI 구현

## 사전 요구사항

- Python 3.8 이상
- Windows 10/11 (다른 OS에서도 동작 가능)

## 설치 및 실행

1. 레포지토리 클론
   ```bash
   git clone <REPO_URL>
   cd pdf_query
   ```
2. 가상환경 생성 및 활성화
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. 의존성 설치
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. 환경변수 설정 (`.env` 파일 수정)
   ```dotenv
   OPENAI_API_KEY=your_openai_api_key
   ```
5. 애플리케이션 실행
   ```bash
   streamlit run app.py
   ```

## 프로젝트 구조

```text
pdf_query/
├── .cache/              # 캐시 디렉토리
│   ├── files/           # 업로드된 PDF 파일 캐시
│   └── embeddings/      # 임베딩 벡터 캐시
├── .vscode/             # VS Code 설정 (선택)
├── prompts/
│   └── pdf-rag.yaml     # Prompt 정의 파일
├── app.py               # 메인 Streamlit 애플리케이션
├── requirements.txt     # 의존성 목록
├── .gitignore           # Git 무시 설정
└── README.md            # 프로젝트 설명 문서
```

## 사용 방법

1. 브라우저에서 `http://localhost:8501` 으로 접속합니다.
2. 사이드바에서 PDF 파일을 업로드합니다.
3. 질문을 입력한 후 대화창에서 답변을 확인합니다.

## 문제 해결

- **TypeError: NoneType** ▶ `prompts/pdf-rag.yaml` 파일 내용을 확인하고 올바른 스펙이 들어 있는지 점검하세요.
- **pip 업그레이드 에러** ▶ `python -m pip install --upgrade pip` 명령을 사용하세요.

## 라이선스

MIT 라이선스. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.
