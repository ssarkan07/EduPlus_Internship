"""
Verification script for EduPlus_Internship Python Environment.
Tests imports and versions across all layers of the ML lifecycle.
"""
import sys

def check_environment():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}\n")
    
    modules = [
        ("Jupyter / Notebook", ["jupyter", "notebook", "IPython"]),
        ("Data Manipulation", ["numpy", "pandas", "scipy"]),
        ("Visualization", ["matplotlib", "seaborn", "plotly"]),
        ("Classical ML", ["sklearn", "xgboost", "lightgbm", "catboost", "joblib"]),
        ("Deep Learning", ["torch", "torchvision", "torchaudio", "transformers", "accelerate"]),
        ("MLOps & Tracking", ["mlflow"]),
        ("Deployment & Serving", ["fastapi", "uvicorn", "pydantic", "gradio", "streamlit"]),
        ("Requested Library", ["gravitas"]),
    ]
    
    all_success = True
    for category, pkgs in modules:
        print(f"--- {category} ---")
        for pkg in pkgs:
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "installed")
                print(f"  [OK] {pkg:<16} (v{ver})")
            except Exception as e:
                print(f"  [FAIL] {pkg:<14} -> {e}")
                all_success = False
        print()
        
    if all_success:
        print(">>> All dependencies installed and imported successfully!")
    else:
        print(">>> Some packages had issues during import.")

if __name__ == "__main__":
    check_environment()
