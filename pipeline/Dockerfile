FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .

# Base image has torch 2.1.0 + CUDA 12.1 pre-installed.
# Filter out torch from requirements to avoid overwriting with CPU version.
RUN pip install --upgrade pip && \
    grep -v '^torch==' requirements.txt > requirements_no_torch.txt && \
    pip install --no-cache-dir -r requirements_no_torch.txt && \
    pip install --no-cache-dir --no-deps "https://files.pythonhosted.org/packages/08/c7/8c33e2660161a99923fa9b46c72c31884efd482b89f7ead970cee5c0072a/lightning-2.1.0-py3-none-any.whl" && \
    pip install --no-cache-dir --no-deps pytorch-forecasting==1.6.1 && \
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import lightning.pytorch as pl; print('lightning.pytorch OK:', pl.__version__)" && \
    python -c "from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH; print('lightning.fabric OK')" && \
    python -c "import pytorch_forecasting; print('pytorch_forecasting OK')"

COPY train.py .
COPY features.py .

ENTRYPOINT ["python", "train.py"]
