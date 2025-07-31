from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

def build_translator_chain(model_name: str = "gpt-4o", temperature: float = 0.0) -> LLMChain:
    """
    질문을 한글과 영어로 번역하는 LLMChain을 반환합니다.
    """
    # 1) 번역용 PromptTemplate
    translate_prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "다음 질문을 자연스러운 영어로 번역해 주세요:\n\n"
            "{text}"
        )
    )

    # 2) ChatOpenAI 인스턴스 (온도 0 고정)
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    # 3) 번역 체인 생성
    translator_chain = LLMChain(
        llm=llm,
        prompt=translate_prompt,
        output_key="translated_question"  # 출력 key 지정
    )

    return translator_chain
