
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements-serve.txt .

RUN pip install --no-cache-dir --timeout=300 --retries=10 -r requirements-serve.txt

# Copy project files
COPY app.py .
COPY model/ ./model/
COPY static/ ./static/

# Expose FastAPI port
EXPOSE 8000

# Start application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
