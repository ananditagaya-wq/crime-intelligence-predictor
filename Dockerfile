FROM python:3.11-slim

WORKDIR /app

COPY requirements-serve.txt .

RUN pip install --no-cache-dir --timeout=300 --retries=10 -r requirements-serve.txt

COPY app.py .
COPY auth/ ./auth/
COPY model/ ./model/
COPY data/crime_dataset_docker.csv ./data/crime_dataset.csv
COPY static/ ./static/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]