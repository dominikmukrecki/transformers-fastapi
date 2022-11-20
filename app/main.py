import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QADataModel(BaseModel):
    question: str
    context: str

from transformers import pipeline
model = pipeline('question-answering')

@app.post("/question_answering")
async def qa(input_data: QADataModel):
    result = model(question = input_data.question, context=input_data.context)
    return result