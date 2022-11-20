import uvicorn
import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QADataModel(BaseModel):
    question: str
    context: str

from transformers import pipeline
pipe = pipeline(model=os.environ['MODEL'])

@app.post("/question_answering")
async def qa(input_data: QADataModel):
    result = pipe(question = input_data.question, context=input_data.context)
    return result