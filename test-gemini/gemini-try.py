from langchain_google_genai import ChatGoogleGenerativeAI

# Import Google API Key
from dotenv import load_dotenv
import os
load_dotenv()

# Set your LLM to Google Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv('GOOGLE_API_KEY'))

response = llm.invoke("Who is Ho Chi Minh?")
print(response.content)