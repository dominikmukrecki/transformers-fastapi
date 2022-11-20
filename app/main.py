import uvicorn

from fastapi import FastAPI

app = FastAPI()

from pydantic import BaseModel

class QADataModel(BaseModel):
    question: str
    context: str

import app.qna