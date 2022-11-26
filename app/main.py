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
    result = util.semantic_search(model.encode(input_data.query), model.encode(input_data.corpus), score_function=util.(os.environ['SENTENCE_SCORE_FUNCTION']))
    return {'query': input_data.query, 'corpus': input_data.corpus, 'result': result, 'model': os.environ['SENTENCE_MODEL']}
