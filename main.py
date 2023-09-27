from fastapi import FastAPI

app = FastAPI()

@app.get("/qa")
async def get_answer(question: str):
    res = qa_chain({"query": question})
    return {"answer": res["answer"]}