import uvicorn

from fastapi import FastAPI

app = FastAPI()

from pydantic import BaseModel

class QADataModel(BaseModel):
    question: str
    context: str

from transformers import pipeline
model_name = 'henryk/bert-base-multilingual-cased-finetuned-polish-squad2'
model = pipeline(model=model_name, tokenizer=model_name, task='question-answering')

@app.post("/question_answering")
async def qa(input_data: QADataModel):
    result = model(question = input_data.question, context=input_data.context)
    return result