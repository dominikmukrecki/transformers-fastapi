import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QADataModel(BaseModel):
    question: str
    context: str

from transformers import pipeline
#model_name = 'azwierzc/herbert-large-poquad'
model_name = AutoTokenizer.from_pretrained("/home/root/.cache/huggingface/transformers/herbert-large-poquad")
model = pipeline(model=model_name, tokenizer=model_name, task='question-answering')

@app.post("/question_answering")
async def qa(input_data: QADataModel):
    result = model(question = input_data.question, context=input_data.context)
    return result