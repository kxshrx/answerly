from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite')

repsonse = model.invoke("what is the reoadmap to learn langchain?")

print(repsonse.content)