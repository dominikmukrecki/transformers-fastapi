import uvicorn
import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SentenceDataModel(BaseModel):
    query: str
    corpus: list

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer(os.environ['SENTENCE_MODEL'])

@app.post('/' + os.environ['SENTENCE_ENDPOINT'])
async def sent(input_data: SentenceDataModel):
    result = util.semantic_search(model.encode(input_data.query), model.encode(input_data.corpus), score_function=util.dot_score)
    return {'result': result[0]}
