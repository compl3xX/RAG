from fastapi import FastAPI

from rag import generate_answer, search_db, record_audio, transcribe_audio, reply, reply_test

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/ask")
def ask():
    record_audio()
    user_query = transcribe_audio()
    chunks = search_db(user_query)
    answer = generate_answer(user_query,chunks)
    reply(answer)
    return {"answer":answer}

@app.get("/test_audio")
def test():
      reply_test()
