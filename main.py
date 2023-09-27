from fastapi import FastAPI
from qa.QARetriever import load_chain


app = FastAPI()
qa_chain = load_chain()

@app.get("/")
def read_root():
    return {"Intro": "OpenAI-Powered Chatbot", "Usage": "GET /qa?question='your_question'"}

@app.get("/qa")
async def get_answer(question: str):
    res = qa_chain({"query": question})
    return {"answer": res["answer"]}