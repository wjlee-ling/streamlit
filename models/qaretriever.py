from typing import Optional, Dict
import langchain
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.cache import InMemoryCache


langchain.llm_cache = InMemoryCache()

class QARetriever:
    # prompt engineering
    template = """Observe the following rules to answer the question at the end.\
    1. Answer the question in a complete sentence.\
    2. Answer in Korean.\
    3. Answer in a polite manner with honorifics. \
    4. If you don't know the answer, just type "잘 모르겠습니다".\
    5. DO NOT swear or use offensive language.\
    Given the rules, the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    chat history: {chat_history}\
    question: {question}\
    answer:"""
    prompt = PromptTemplate.from_template(template)

    def __init__(self, url:str, model_name:str="gpt-4", condense_question_llm:Optional[str]=None):
        if condense_question_llm is None:
            condense_question_llm = ChatOpenAI(
                temperature=0,
                model_name=model_name,
            )            
        # extract (distill) the retrieved documents into an answer using LLM/Chat model
        self.llm = ChatOpenAI(
            model_name=model_name, # GPT-4는 chat_model 사용
            temperature=0,
        )

        # memory for chat history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

        loader = WebBaseLoader(url)
        data = loader.load()
        text_spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
        splits = text_spliter.split_documents(data)
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=OpenAIEmbeddings(), 
            collection_metadata={"hnsw:space":"cosine"}, # default is "l2" (https://docs.trychroma.com/usage-guide)
        )
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            # chain_type: 
            # "stuff": default; to use all of the text from the documents in the prompt
            # "map_reduce": to batchify docs and feeds each batch with the question to LLM, and come up with the final answer based on the answers
            # "refine": to batchify docs and feeds the first batch to LLM, and then feeds the second batch with the answer from the first one, and so on
            # "map-rerank": to batchify docs and feeds each batch, return a score and come up with the final answer based on the scores
            self.llm,
            retriever=vectorstore.as_retriever(), 
            # chain_type_kwargs={"prompt": self.prompt}, 
            memory=self.memory,
            condense_question_llm=condense_question_llm,
            condense_question_prompt=self.prompt,
        )

    def __call__(self, query:str) -> Dict[str, str]:
        import time
        start_time = time.time()
        res = self.qa_chain({"question": query})
        end_time = time.time()
        print(end_time - start_time)
        return res

if __name__ == "__main__":
    import argparse 

    parser = argparse.ArgumentParser(description='Run a basic QA Retriever powered by ChatGPT-4')
    parser.add_argument('-q', '--question', type=str, required=True)
    parser.add_argument('-u', '--url', type=str, default="https://textnet.kr/about")

    args = parser.parse_args()
    bot = QARetriever(url=args.url)
    
    res = bot(query=args.question) # {"query": args.question}
    print(res)