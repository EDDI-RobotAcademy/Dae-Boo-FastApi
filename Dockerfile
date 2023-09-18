FROM arm64v8/python:3.7

COPY ./app /app/app
COPY requirements.txt /app
WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 3002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3002"]
