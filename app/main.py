import uvicorn
import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

#class QNADataModel(BaseModel):
#    question: str
#    context: str

#from transformers import pipeline
#pipe = pipeline(model=os.environ['QNA_MODEL'])

#@app.post('/' + os.environ['QNA_ENDPOINT'])
#async def qa(input_data: QNADataModel):
#    result = pipe(question = input_data.question, context=input_data.context)
#    return result

class SentenceDataModel(BaseModel):
    query: str
    corpus: list

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer(os.environ['SENTENCE_MODEL'])

@app.post('/' + os.environ['SENTENCE_ENDPOINT'])
async def sent(input_data: SentenceDataModel):
    result = util.semantic_search(model.encode(input_data.query), model.encode(input_data.corpus), top_k=1)
    return result[0].corpus_id
