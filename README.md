
# Automated Classification of Second and Third Degree Burn Images

This repository contains code, trained models, experiment artifacts and notebooks used to develop and evaluate convolutional neural networks for classifying second and third degree burn images.


## Setup
1. Create a Python `3.11.9` environment (recommended: venv or conda).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the demo interface
The Gradio demo is in `interface/gradio_burn_classification.py`. To run it locally:

```bash
python interface/gradio_burn_classification.py
```

Notes:
- The script expects a trained model file at `models/proposed_model/burn_green_cnn.keras` by default — update the path in the script if you want to use a different model from `models/`.
- The model used by the Gradio app preprocesses images to size 540×960 and uses only the green channel (single-channel input).

## Project structure

- `requirements.txt` — pinned Python dependencies used in experiments and the interface.
- `LICENSE` — project license file.
- `data/`
  - `results/` — CSV outputs and exports from experiments and analysis. Example files found include: `channel_analysis.csv`, `hyperparameter_search.csv`, `model_metrics.csv`, `wandb_export.csv`.
  - (Note) Raw datasets (images) are not included in this repository — provide training/validation/test folders with names such as `second_degree/` and `third_degree/` when running the notebooks or training scripts.
- `figures/` — generated plots and figures used in the manuscript and analysis.
- `interface/` — minimal Gradio-based inference UI: `gradio_burn_classification.py`.
- `models/` — saved model checkpoints in Keras format. Examples in this repository include `mobilenetv2.keras`, `resnet50.keras`, `vgg16.keras`, and `burn_green_cnn.keras`. There may be subfolders (e.g., `proposed_model/`) which contain the model used by the Gradio demo.
- `notebooks/` — Jupyter notebooks for exploration, evaluation and figure generation. Notable files:
  - `best_model_green_channel.ipynb` — proposed architecture in the manuscript that uses only the green channel.
  - `transfer_learning_training.ipynb` — transfer-learning experiments and setup notes.
  - `model_test_evaluation.ipynb` — aggregated model evaluation and metric calculations.
  - `hyperparameter_search_wandb.ipynb` — hyperparameter search analysis and W&B integration.
  - `wandb_csv_preprocessing.ipynb` — scripts to preprocess W&B CSV exports.
  - `figure_plots.ipynb` — generation of the figures included in the manuscript.
  - `tools.py` — helper utilities used across notebooks (image loading, resizing, evaluation helpers).

## Model details and inference
- Expected input shape: images are resized to 540 (width) × 960 (height) in the preprocessing pipeline used by the notebooks and the Gradio app.
- Channel handling: several experiments use the full RGB image, while some models (including the Gradio demo model) use only the green channel (single-channel input). See `notebooks/tools.py` for the data-loading helper `load_images_with_labels()` and details about channel selection.
- Loading a Keras model in code:

```python
from tensorflow.keras.models import load_model
model = load_model('models/proposed_model/burn_green_cnn.keras')

# Example prediction pipeline (consistent with the repo's preprocessing):
import cv2, numpy as np
img = cv2.imread('path/to/image.jpg')
resized = cv2.resize(img, (540, 960))
green = resized[:, :, 1]
X = np.expand_dims(green, axis=(0, -1))
pred = model.predict(X)
```

## Notebooks and analysis
- Use the notebooks in `notebooks/` to reproduce experiments, plots and evaluation metrics. The helper functions in `notebooks/tools.py` include `load_images_with_labels()` and `evaluate_model()` which are used across analysis notebooks.

## Experiments and tracking
- Weights & Biases (W&B) was used for experiment tracking; `data/results/wandb_export.csv` contains exported run data used for post-hoc analysis.


## License
- See `LICENSE` for license terms.

## Contact
- Yamile Montecinos Rodríguez - [LinkedIn](https://www.linkedin.com/in/yamilemontecinos/)
- Francisco Javier Torres Santana - [LinkedIn](https://www.linkedin.com/in/franjtorress/)
- Lorena Díaz González - [ResearchGate](https://www.researchgate.net/profile/Lorena-Diaz-Gonzalez)
