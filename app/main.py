import uvicorn
import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QNADataModel(BaseModel):
    question: str
    context: str

from transformers import pipeline
pipe = pipeline(model=os.environ['QNA_MODEL'])

@app.post(os.environ['QNA_ENDPOINT'])
async def qa(input_data: QNADataModel):
    result = pipe(question = input_data.question, context=input_data.context)
    return result