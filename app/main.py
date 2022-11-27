import uvicorn
import os
from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

class ScoreFunction(str, Enum):
    cos_sim = util.cos_sim
    dot_score = uti.dot_score

class SentenceDataModel(BaseModel):
    query: str
    corpus: list
    top_k: int
    score_function: ScoreFunction
    class Config:
        use_enum_values = True

model = SentenceTransformer(os.environ['SENTENCE_MODEL'])
model.max_seq_length = int(os.environ['SENTENCE_MODEL_MAX_SEQ_LENGTH'])

@app.post('/' + os.environ['SENTENCE_ENDPOINT'])
async def sent(input_data: SentenceDataModel):
    result = util.semantic_search(model.encode(input_data.query), model.encode(input_data.corpus), score_function=input_data.score_function, top_k=input_data.top_k)
    return {'input_data': input_data, 'result': result, 'model': os.environ['SENTENCE_MODEL']}
