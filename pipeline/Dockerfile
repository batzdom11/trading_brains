FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Install lightning 2.1.0 directly from wheel URL (package is quarantined on PyPI but files still hosted)
    pip install --no-cache-dir --no-deps "https://files.pythonhosted.org/packages/08/c7/8c33e2660161a99923fa9b46c72c31884efd482b89f7ead970cee5c0072a/lightning-2.1.0-py3-none-any.whl" && \
    pip install --no-cache-dir --no-deps pytorch-forecasting==1.6.1 && \
    # Verify imports work
    python -c "import lightning.pytorch as pl; print('lightning.pytorch OK:', pl.__version__)" && \
    python -c "from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH; print('lightning.fabric OK')" && \
    python -c "import pytorch_forecasting; print('pytorch_forecasting OK')"

COPY train.py .
COPY features.py .

ENTRYPOINT ["python", "train.py"]
