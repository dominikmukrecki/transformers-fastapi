FROM python:3.7

RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install python-dotenv pydantic fastapi uvicorn transformers sentence-transformers

COPY ./app /app