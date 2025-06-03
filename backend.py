from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import create_llm_chain, QAResponse

app = FastAPI()

# === Input model ===
class QuestionRequest(BaseModel):
    question: str
    url: str  # URL of the webpage the user is asking about

# === Health check ===
@app.get("/")
async def working():
    return {"message": "Backend is running!"}

# === Main POST route ===
@app.post("/ask", response_model=QAResponse)
async def ask_question(request: QuestionRequest):
    try:
        response = create_llm_chain(question=request.question, url=request.url)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
