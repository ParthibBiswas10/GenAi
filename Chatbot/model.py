from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

# Force load .env
load_dotenv()



llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.6",
    task="text-generation",
    temperature=0.1,

)

model = ChatHuggingFace(llm=llm)
history=[]
while True:
    user=input("You: ")
    history.append(HumanMessage(content=user))
    if user=='exit':
        break
    result=model.invoke(history)
    history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print("Exiting chat...gm")
print(history)
