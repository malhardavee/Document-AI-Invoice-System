FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY ./models ./models

EXPOSE 8000

CMD ["uvicorn", "src.api_app:app", "--host", "0.0.0.0", "--port", "8000"]
