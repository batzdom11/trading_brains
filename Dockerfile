FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir --upgrade -r requirements.txt

COPY main.py .

CMD ["python", "main.py"]