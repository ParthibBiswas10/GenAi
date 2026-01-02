from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    
)

template1=PromptTemplate(
    input_variables=["topic"],
    template="Provide a detailed summary of the following topic: {topic}",
)

template2=PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text: {text}",
)

prompt1=template1.format(topic="messi")
result1=llm.invoke(prompt1)

prompt2=template2.format(text=result1.content)
result2=llm.invoke(prompt2)
print("Final Summary: ", result2.content)
