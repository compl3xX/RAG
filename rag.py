from faiss import IndexFlatL2
from langchain_community.docstore import InMemoryDocstore
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from pydantic import SecretStr
from dotenv import load_dotenv
import os


# Load variables from .env into the environment
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# Load documents
loader = PyPDFLoader(r"C:\Users\rishu\Downloads\Essential-GraphRAG.pdf")
docs = loader.load()

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

# Step 2: Add documents
vector_store.add_documents(docs)  # <- docs must be list of Document objects


# Step 3: Search function
def search_db(query: str, k: int = 3):
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


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

