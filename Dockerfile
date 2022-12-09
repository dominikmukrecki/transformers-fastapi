FROM python:3.7

RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install pydantic fastapi uvicorn sentence-transformers huggingface_hub transformers

COPY ./app /app