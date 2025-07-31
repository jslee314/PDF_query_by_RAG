class HybridRetriever:
    """
    벡터 + 키워드 검색 폴백 하이브리드 리트리버
    """
    def __init__(self, vector_retriever, docs):
        self.vec = vector_retriever
        self.docs = docs

    def get_relevant_documents(self, query: str):
        vec_docs = self.vec.get_relevant_documents(query)
        kw_docs  = [d for d in self.docs if query in d.page_content]
        # 중복 제거
        combined = {id(d): d for d in vec_docs + kw_docs}
        return list(combined.values())