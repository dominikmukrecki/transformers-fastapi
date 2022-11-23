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

query_embedding = model.encode('A man is eating pasta.')
passage_embedding = model.encode(['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.'
          ])

class SentenceAsker(BaseModel):
    question: str
#    context: str


@app.post('/sentence1')
async def sent(input_data: SentenceAsker):
    result = zip(util.semantic_search(query_embedding, passage_embedding).sort(key=corpus_id), passage_embedding)
    return result
