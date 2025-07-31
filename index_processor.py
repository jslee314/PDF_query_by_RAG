import streamlit as st
from langchain.callbacks.tracers import LangChainTracer
from rag.loader import load_and_split
from rag.embedder import get_embeddings, create_vectorstore
from rag.retriever import HybridRetriever
from rag.translator import build_translator_chain

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import yaml


def create_chain(retriever_fn, model_name="gpt-4o", temperature=0.0, tracer=None):
    with open("prompts/pdf-rag.yaml", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    prompt = PromptTemplate(
        template=spec["template"],
        input_variables=spec["input_variables"],
    )
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        callbacks=[tracer] if tracer else None
    )
    return (
        {"context": retriever_fn, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def render(uploaded_files, selected_model, temperature, translator_chain, tracer):
    st.header("1. 문서 업로드 및 처리")

    if not uploaded_files:
        st.info("먼저 PDF 파일을 업로드해 주세요.")
        return

    total = len(uploaded_files)
    progress = st.progress(0)

    split_docs_by_file = {}
    file_retrievers = {}
    file_chains = {}

    for idx, f in enumerate(uploaded_files, start=1):
        st.subheader(f"🔄 처리 중: {f.name}")
        docs = load_and_split([f])
        st.write(f"- 청크 생성: {len(docs)}개")
        split_docs_by_file[f.name] = docs

        embeddings = get_embeddings()
        st.write("- 임베딩 객체 생성 완료")

        vs = create_vectorstore(docs, embeddings)
        n_vectors = getattr(vs.index, "ntotal", "N/A")
        st.write(f"- 벡터스토어 생성 완료: 총 {n_vectors}개 벡터")
        file_retrievers[f.name] = vs.as_retriever()

        chain = create_chain(
            retriever_fn=file_retrievers[f.name],
            model_name=selected_model,
            temperature=temperature,
            tracer=tracer
        )
        st.write("- RAG 체인 초기화 완료")
        file_chains[f.name] = chain

        progress.progress(int(idx / total * 100))

    progress.empty()
    st.success("✅ 문서 처리 완료!")

    st.session_state["split_docs_by_file"] = split_docs_by_file
    st.session_state["file_retrievers"] = file_retrievers
    st.session_state["file_chains"] = file_chains
    st.session_state["processed"] = True