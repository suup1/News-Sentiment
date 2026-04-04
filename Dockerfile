FROM python:3.9

WORKDIR /app

# Copy all files
COPY . .

# Set Python path (fix imports)
ENV PYTHONPATH=/app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run FastAPI
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]