# EduPlus_Internship

Machine Learning workspace setup with an end-to-end environment from data processing and model training to deployment, interactive exploration, and serving.

---

## 1. Virtual Environment Activation

The virtual environment is created at `venv/`.

### PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
*(If script execution is disabled, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

### Command Prompt (CMD):
```cmd
venv\Scripts\activate.bat
```

### Git Bash:
```bash
source venv/Scripts/activate
```

---

## 2. Using Jupyter Notebook

The environment is registered with `ipykernel` under the name **`eduplus_env`** (displayed as `"Python (EduPlus_Internship)"`).

- **Launch Jupyter Lab:**
  ```powershell
  jupyter lab
  ```
- **Launch classic Notebook:**
  ```powershell
  jupyter notebook
  ```
- **Inside VS Code:**
  Open any `.ipynb` file and select the kernel: **Python (EduPlus_Internship)** (`./venv/Scripts/python.exe`).

---

## 3. Installed Stack Overview

| Category | Libraries |
| :--- | :--- |
| **Interactive Development** | `jupyter`, `notebook`, `ipykernel`, `ipywidgets` |
| **Data Processing & Math** | `numpy`, `pandas`, `scipy`, `openpyxl` |
| **Data Visualization** | `matplotlib`, `seaborn`, `plotly` |
| **Classical Machine Learning** | `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `joblib` |
| **Deep Learning & NLP** | `torch`, `torchvision`, `torchaudio`, `transformers`, `datasets`, `accelerate` |
| **MLOps & Tracking** | `mlflow` |
| **Model Deployment & APIs** | `fastapi`, `uvicorn`, `pydantic`, `httpx`, `requests` |
| **Interactive ML Web UIs** | `gradio`, `streamlit` |
| **Astrodynamics / Framework** | `gravitas` |

---

## 4. Verification

To verify that all dependencies are working properly:
```powershell
.\venv\Scripts\python.exe verify_env.py
```
