FROM python:3.7

RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install pydantic fastapi uvicorn transformers sentence-transformers protobuf==3.20.3 sacremoses

COPY ./app /app