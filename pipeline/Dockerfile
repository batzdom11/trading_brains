FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps pytorch-forecasting==1.6.1 && \
    # Create 'lightning' shim package so `import lightning.pytorch` works
    # pytorch_forecasting 1.6.1 uses `import lightning.pytorch as pl` internally
    SITE=$(python -c "import site; print(site.getsitepackages()[0])") && \
    mkdir -p "$SITE/lightning" && \
    printf 'import pytorch_lightning\nimport sys\nsys.modules["lightning.pytorch"] = pytorch_lightning\n' > "$SITE/lightning/__init__.py" && \
    ln -s "$SITE/pytorch_lightning" "$SITE/lightning/pytorch" && \
    python -c "import lightning.pytorch as pl; print('lightning shim OK:', pl.__version__)"

COPY train.py .
COPY features.py .

ENTRYPOINT ["python", "train.py"]
