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
import uuid

# === Load environment ===
load_dotenv()
DEV_MODE = os.getenv("DEV_MODE", "true").strip().lower() == "true"

# === Define Pydantic model for structured output ===
class QAResponse(BaseModel):
    answer: str
    source: str  # "Based on webpage content" or "Generated from general knowledge"

# === Initialize LLM and embedding model ===
llm_model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite')
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Prompt Template ===
prompt_template = PromptTemplate(
    template=(
        "You are an intelligent assistant designed to answer questions strictly based on the content of a given webpage.\n"
        "Use the following extracted content from the page to provide accurate, concise, and clear answers.\n"
        "If the answer is not explicitly found in the provided context, you may use your general knowledge but please clearly indicate that. and dont hesistate from using your knowledge base if answer is absent in the provided context.\n\n"
        "Webpage Content:\n{context}\n\n"
        "User Question:\n{query}\n\n"
        "Please respond with a JSON object having two fields:\n"
        "1. 'answer': your answer as a string.\n"
        "2. 'source': indicate if the answer is 'Based on webpage content' or 'Generated from general knowledge'.\n\n"
        "JSON Response:"
    ),
    input_variables=["context", "query"]
)

# === Output Parser ===
output_parser = PydanticOutputParser(pydantic_object=QAResponse)


# === Main dynamic function ===
def create_llm_chain(question: str, url: str) -> QAResponse:
    # 1. Load webpage content
    loader = UnstructuredURLLoader(urls=[url])
    try:
        documents = loader.load()
        if not documents:
            raise ValueError("No content could be extracted from the URL.")
    except Exception as e:
        raise RuntimeError(f"Failed to load content from {url}: {str(e)}")

    # 2. Split into text chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = [chunk for doc in documents for chunk in text_splitter.split_text(doc.page_content)]

    if not chunks:
        raise RuntimeError("No text chunks generated from the webpage.")

    # 3. Build vector store
    collection_name = f"qa_session_{uuid.uuid4().hex}" if DEV_MODE else "qa_db"
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=None if DEV_MODE else "chroma_db"
    )
    if not DEV_MODE:
        vector_store.persist()

    # 4. Define similarity search function
    def vector_search_fn(query: str) -> str:
        embedded_query = embedding_model.embed_query(query)
        docs = vector_store.similarity_search_by_vector(embedding=embedded_query, k=5)
        return "\n\n".join([doc.page_content for doc in docs])

    # 5. Construct pipeline
    chain = (
        RunnableParallel({
            "context": RunnableLambda(vector_search_fn),
            "query": RunnablePassthrough()
        })
        | prompt_template
        | llm_model
        | StrOutputParser()
        | output_parser
    )

    return chain.invoke(question)


# === CLI for local testing ===
if __name__ == "__main__":
    question = input("Enter your question: ").strip()
    url = input("Enter the URL: ").strip()

    try:
        print("\nFetching answer...")
        response = create_llm_chain(question, url)
        print("\n=== RESPONSE ===")
        print("Answer:", response.answer)
        print("Source:", response.source)
    except Exception as e:
        print("Error:", e)
