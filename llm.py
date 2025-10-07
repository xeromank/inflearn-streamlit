from dotenv import load_dotenv
from langchain import hub
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm


def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해 줍니다.
        사전 : {dictionary}
    
        질문 : {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain


def get_retriever():
    embedding = OpenAIEmbeddings(model = "text-embedding-3-large")
    index_name = 'tax-markdown-index3'
    database = PineconeVectorStore.from_existing_index(embedding=embedding, index_name=index_name)
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever


def get_qa_chain():
    llm = get_llm()
    retriever = get_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    return RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )


def get_ai_message(user_message):
    load_dotenv()

    qa_chain = get_qa_chain()

    tax_chain = {"query": get_dictionary_chain()} | qa_chain
    ai_message = tax_chain.invoke({"question": user_message})
    return ai_message["result"]
