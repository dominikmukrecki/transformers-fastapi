import uvicorn
import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SentenceDataModel(BaseModel):
    query: str
    corpus: list
    top_k: int
    score_function: str

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer(os.environ['SENTENCE_MODEL'])

@app.post('/' + os.environ['SENTENCE_ENDPOINT'])
async def sent(input_data: SentenceDataModel):
    if input_data.score_function == 'dot_score':
        score_function = util.dot_score
    elif input_data.score_function == 'cos_sim':
        score_function = util.cos_sim
    else:
        return null
    result = util.semantic_search(model.encode(input_data.query), model.encode(input_data.corpus), score_function=util.dot_score, top_k=input_data.top_k)
    return {'query': input_data.query, 'corpus': input_data.corpus, 'result': result, 'model': os.environ['SENTENCE_MODEL']}
