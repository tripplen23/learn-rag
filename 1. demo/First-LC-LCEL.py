# This code is the demonstrate a simple way of forming a Prompt and using it to Chain with a Model

from langchain_core.prompts import ChatPromptTemplate
#from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

def main():

  print(demosimple.__doc__)
  demosimple()

def demosimple():
  """
  This Function Demonstrates a simple use of LCEL (Langchain Expression Language) to create a custom Chain with the Prompt and Model
  """

  # Create the Prompt Template
  prompt = ChatPromptTemplate.from_template("Tell me a few achievements of {name}")

  # Create the LLM Object
  #model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.5)
  model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.5)

  # Create the Chain
  chain = prompt | model # LCEL - Langchain Expression Language

  # Invoke (run) the Chain - The Chat model returns a message
  print(chain.invoke({"name": "Elon Musk"}).content)

if __name__ == "__main__":
    main()  
