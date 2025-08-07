from faiss import IndexFlatL2
from langchain.retrievers import EnsembleRetriever
from langchain_community.docstore import InMemoryDocstore
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from pydantic import SecretStr
from dotenv import load_dotenv
import os


# Load variables from .env into the environment
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# Load documents
loader = PyPDFLoader(r"C:\Users\rishu\Downloads\Essential-GraphRAG.pdf")
docs = loader.load()

# Split based on sections/headers using recursive logic
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],
)

chunks = splitter.split_documents(docs)

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Encode and build FAISS index with LangChain wrapper
dimension = model.get_sentence_embedding_dimension()

index = IndexFlatL2(dimension)

def embedding_function(text: str) -> list[float]:
    return model.encode([text])[0].tolist()

vector_store = FAISS(
    embedding_function=embedding_function,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Step 2: Add documents to vector DB
vector_store.add_documents(chunks)  # <- docs must be list of Document objects

# Step 2.1: Add documents to bm25
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

vector_retriever =  vector_store.as_retriever(search_kwargs={"k": 3,"score_threshold":0.80})

# Collate the retriever
hybrid_retriever =  EnsembleRetriever(
    retrievers=[bm25_retriever,vector_retriever],
    weights= [0.3,0.7]
)

rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query,passages):
    pairs = [(query,passage.page_content) for passage in passages]
    scores = rerank_model.predict(pairs)
    ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return ranked

# Step 3: Search function
def search_db(query: str, k: int = 3):
    results = hybrid_retriever.invoke(query)
    results = rerank(query,results)
    return [passage.page_content for passage,scores in results if scores > 0]


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    api_key = SecretStr(api_key),
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question based only on the context provided.If the context is not relevant, say you don't know."),
    ("user", "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
])


parser = StrOutputParser()

chain = prompt | llm | parser


def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n".join(context_chunks)
    return chain.invoke({"query": query, "context": context})

