from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import sys

# === Load environment ===
load_dotenv()
DEV_MODE = os.getenv("DEV_MODE", "true").strip().lower() == "true"

# === Define Pydantic model for structured output ===
class QAResponse(BaseModel):
    answer: str
    source: str  # "Based on webpage content" or "Generated from general knowledge"

# === Initialize LLM model ===
llm_model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite')

# === Embedding model ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Load webpage content ===
URLS = ['https://en.wikipedia.org/wiki/FIFA_World_Cup']
loader = UnstructuredURLLoader(urls=URLS)

try:
    documents = loader.load()
    if not documents:
        raise ValueError("No content was extracted from the webpage.")
except Exception as e:
    print(f"Error fetching or processing {URLS[0]}, exception: {e}")
    documents = []

if not documents:
    print("No documents to process. Exiting...")
    sys.exit(1)

# === Split text ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = [chunk for doc in documents for chunk in text_splitter.split_text(doc.page_content)]

if not chunks:
    print("No text chunks generated. Exiting...")
    sys.exit(1)

# === Build vector store ===
vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    collection_name="qa_db",
    persist_directory=None if DEV_MODE else "chroma_db"
)
if not DEV_MODE:
    vector_store.persist()

# === Prompt template ===
prompt_template = PromptTemplate(
    template=(
        "You are an intelligent assistant designed to answer questions strictly based on the content of a given webpage.\n"
        "Use the following extracted content from the page to provide accurate, concise, and clear answers.\n"
        "If the answer is not explicitly found in the provided context, you may use your general knowledge but please clearly indicate that.\n\n"
        "Webpage Content:\n{context}\n\n"
        "User Question:\n{query}\n\n"
        "Please respond with a JSON object having two fields:\n"
        "1. 'answer': your answer as a string.\n"
        "2. 'source': indicate if the answer is 'Based on webpage content' or 'Generated from general knowledge'.\n\n"
        "JSON Response:"
    ),
    input_variables=["context", "query"]
)

# === Output parser ===
output_parser = PydanticOutputParser(pydantic_object=QAResponse)

# === Vector search chain ===
def vector_search_fn(query: str) -> str:
    embedded_query = embedding_model.embed_query(query)
    docs = vector_store.similarity_search_by_vector(embedding=embedded_query, k=5)
    return "\n\n".join([doc.page_content for doc in docs])

vector_search_chain = RunnableLambda(vector_search_fn)

# === Prompt construction chain (parallel + prompt fill) ===
parallel_chain = RunnableParallel({
    "context": vector_search_chain,
    "query": RunnablePassthrough()
})

# === Final chain: prompt → LLM → parse ===
llm_chain = (
    parallel_chain
    | prompt_template
    | llm_model
    | StrOutputParser()
    | output_parser
)

# === Execution ===
if __name__ == "__main__":
    question = "How does the cricket World Cup tournament work?"
    try:
        response = llm_chain.invoke(question)
        print("\nAnswer:", response.answer)
        print("Source:", response.source)
    except Exception as e:
        print("Failed to generate response:", e)
