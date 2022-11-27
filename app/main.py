import uvicorn
import os
from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

app = FastAPI()

# semantic search
if os.environ['SEMANTIC_SEARCH_MODEL'] is not None:
    class ScoreFunction(str, Enum):
        cos_sim = 'cos_sim'
        dot_score = 'dot_score'

    class SemanticSearchDataModel(BaseModel):
        query: str
        corpus: list
        top_k: int
        score_function: ScoreFunction
        class Config:
            use_enum_values = True

    semantic_search_model = SentenceTransformer(os.environ['SEMANTIC_SEARCH_MODEL'])
    semantic_search_model.max_seq_length = int(os.environ['SEMANTIC_SEARCH_MODEL_MAX_SEQ_LENGTH'])

    @app.post('/' + os.environ['SEMANTIC_SEARCH_ENDPOINT'])
    async def sent(input_data: SemanticSearchDataModel):
        if input_data.score_function == 'dot_score':
            score_function = util.dot_score
        elif input_data.score_function == 'cos_sim':
            score_function = util.cos_sim
        else:
            score_function = None
        result = util.semantic_search(semantic_search_model.encode(input_data.query), semantic_search_model.encode(input_data.corpus), score_function=score_function, top_k=input_data.top_k)
        return {'input_data': input_data, 'result': result, 'model': os.environ['SEMANTIC_SEARCH_MODEL']}

# zero-shot classification

if os.environ['ZERO_SHOT_CLASSIFICATION_ENDPOINT'] is not None:

    class ClassificationDataModel(BaseModel):
        sequence: str
        labels: list
        multi_label = False

    zero_shot_classification_model = pipeline('zero-shot-classification', model="facebook/bart-large-mnli")

    @app.post('/' + os.environ['ZERO_SHOT_CLASSIFICATION_ENDPOINT'])
    async def sent(input_data: ClassificationDataModel):
        result = zero_shot_classification_model(input_data.sequence, input_data.labels, multi_label=input_data.multi_label)
        return {'result': result, 'multi_label' : input_data.multi_label, 'model': os.environ['ZERO_SHOT_CLASSIFICATION_MODEL']}