import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')



current_file_dir = Path(__file__).parent

yaml_file_path = current_file_dir/ "system_prompt.yaml"

resolved_yaml_file_path = yaml_file_path.resolve()


with open(resolved_yaml_file_path, "r") as f:
    prompt = yaml.safe_load(f)

prompt = prompt["system_prompt"]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""{prompt}""",
        ),
        ("human", "{input}"),
    ]
)

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    api_key = groq_api_key
)



class Query(BaseModel):
    user_query : str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query/")
async def process_query(request: Query):
    chain = prompt | llm
    response = chain.invoke(
    {
       
        "input": request.user_query,
    }
    )


    return {"answer":response.content}
