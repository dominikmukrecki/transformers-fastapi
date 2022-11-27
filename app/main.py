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
model.max_seq_length = int(os.environ['SENTENCE_MODEL_MAX_SEQ_LENGTH'])

@app.post('/' + os.environ['SENTENCE_ENDPOINT'])
async def sent(input_data: SentenceDataModel):
    if input_data.score_function == 'dot_score':
        score_function = util.dot_score
    elif input_data.score_function == 'cos_sim':
        score_function = util.cos_sim
    else:
        score_function = null
    result = util.semantic_search(model.encode(input_data.query), model.encode(input_data.corpus), score_function=score_function, top_k=input_data.top_k)
    return {'input_data': input_data, 'result': result, 'model': os.environ['SENTENCE_MODEL']}
