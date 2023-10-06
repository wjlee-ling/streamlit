from models import QARetriever

from typing import Union
from fastapi import FastAPI

app = FastAPI()
bot = QARetriever(url="https://textnet.kr/about")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/bot")
def chat(query: Union[str, None] = None):
    if query is None:
        return {"error": "No query received"}
    res = bot(query)
    return res
