.PHONY: install train app docker-build docker-train docker-app clean

VENV?=.venv

install:
	python -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

train:
	python train_model.py

app:
	streamlit run app.py --server.port=8501

docker-build:
	docker build -t cognitive-soar:latest .

docker-train:
	docker run --rm -v $(PWD)/models:/app/models -v $(PWD)/reports:/app/reports -v $(PWD)/data:/app/data cognitive-soar:latest python train_model.py

docker-app:
	docker run --rm -p 8501:8501 -v $(PWD)/models:/app/models cognitive-soar:latest streamlit run app.py --server.port=8501 --server.address=0.0.0.0

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache .mypy_cache *.log *.png
