from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Force load .env
load_dotenv()



llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.6",
    task="text-generation",
    

)

model = ChatHuggingFace(llm=llm)
history=[]
while True:
    user=input("You: ")
    history.append(user)
    if user=='exit':
        break
    result=model.invoke(history)
    history.append(result.content)
    print("AI: ",result.content)

