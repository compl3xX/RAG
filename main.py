from fastapi import FastAPI

from rag import generate_answer, search_db

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/ask")
def ask(query: str):
    chunks = search_db(query)
    answer = generate_answer(query,chunks)
    return {"answer":answer}
