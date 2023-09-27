import argparse
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


loader = WebBaseLoader("https://textnet.kr/about")
data = loader.load()
# loaders = [....]
# data = []
# for loader in loaders:
#     data.extend(loader.load())

text_spliter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=0)
splits = text_spliter.split_documents(data)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# extract (distill) the retrieved documents into an answer using LLM/Chat model
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
)

# prompt engineering
template = """Observe the following rules to answer the question at the end.\
1. Answer the question in a complete sentence.\
2. Answer in Korean.\
3. Answer in a polite manner with honorifics. \
4. If you don't know the answer, just type "잘 모르겠습니다".\
5. DO NOT swear or use offensive language.\
{context}\
question: {question}\
answer:"""
prompt = PromptTemplate.from_template(template)

def load_chain():
    qa_chain = RetrievalQA.from_chain_type(
        # chain_type: 
        # "stuff": default; to use all of the text from the documents in the prompt
        # "map_reduce": to batchify docs and feeds each batch with the question to LLM, and come up with the final answer based on the answers
        # "refine": to batchify docs and feeds the first batch to LLM, and then feeds the second batch with the answer from the first one, and so on
        # "map-rerank": to batchify docs and feeds each batch, return a score and come up with the final answer based on the scores
        llm,
        retriever=vectorstore.as_retriever(), 
        chain_type_kwargs={"prompt": prompt}, 
    )
    return qa_chain

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a basic QA Retriever powered by ChatGPT-4')
    parser.add_argument('-q', '--question', type=str, required=True)

    args = parser.parse_args()
    question = args.question
    qa_chain = load_chain()
    res = qa_chain({"query": question})
    print(res)