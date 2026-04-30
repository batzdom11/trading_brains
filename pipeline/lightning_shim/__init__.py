# Shim: make `import lightning.pytorch` resolve to `pytorch_lightning`
# Required because pytorch_forecasting 1.6.1 imports from `lightning.pytorch`
# but the `lightning` package is quarantined on PyPI.
import pytorch_lightning as pytorch
from pytorch_lightning import *
