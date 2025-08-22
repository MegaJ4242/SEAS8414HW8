# INSTALL

## Local (Windows Git Bash)

```bash
cd "/c/Users/<YOU>/Desktop/cognitive-soar"
py -3.11 -m venv .venv || python -m venv .venv
source .venv/Scripts/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python train_model.py
streamlit run app.py
```
