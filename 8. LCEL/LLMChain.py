# This program demonstrate a simple usage of creating a LCEL based chain
# The chain comprises a prompt, thw llm object and a Stringoutput parser

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

llm = GoogleGenerativeAI(
  model="gemini-1.5-pro-latest",
  temperature=0.5,
  google_api_key=os.getenv("GOOGLE_API_KEY"), 
)

prompt = ChatPromptTemplate.from_messages([
  ("system", "You are world class technical documentation writer."),
  ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

output = chain.invoke({"input": "How can langsmith help with testing?"})

print(output)