FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps pytorch-forecasting==1.6.1

COPY train.py .
COPY features.py .

ENTRYPOINT ["python", "train.py"]
