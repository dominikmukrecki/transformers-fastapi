FROM python:3.7

RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install pydantic fastapi uvicorn transformers

RUN git lfs install

EXPOSE 8000

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
