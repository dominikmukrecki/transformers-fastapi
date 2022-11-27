FROM python:3.7

RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install pydantic fastapi uvicorn transformers sentence-transformers sacremoses protobuf==3.20.3

COPY ./app /app