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

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

query_embedding = model.encode('How big is London')
passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
                                  'London is known for its finacial district'])

class SentenceAsker(BaseModel):
    question: str
#    context: str


@app.post('/sentence')
async def sent(input_data: SentenceAsker):
    result = util.dot_score(query_embedding, passage_embedding)
    return {"test": result}
