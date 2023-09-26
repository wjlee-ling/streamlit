import argparse
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


loader = WebBaseLoader("https://textnet.kr/about")
data = loader.load()

text_spliter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=0)
splits = text_spliter.split_documents(data)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# extract (distill) the retrieved documents into an answer using LLM/Chat model
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a basic QA Retriever powered by ChatGPT-4')
    parser.add_argument('-q', '--question', type=str, required=True)

    args = parser.parse_args()
    question = args.question
    res = qa_chain({"query": question})
    print(res)