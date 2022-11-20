FROM python:3.7

RUN pip install tensorflow-cpu

RUN pip install pydantic fastapi uvicorn transformers

EXPOSE 80

EXPOSE 8000

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]