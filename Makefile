.PHONY: help install api web test showcase clean

help:
	@echo "Duat Local Orchestration Commands:"
	@echo "  make install  - Install Python backend dependencies"
	@echo "  make api      - Start the FastAPI backend server on port 8000"
	@echo "  make web      - Start the Vite React frontend dev server on port 5173"
	@echo "  make test     - Run backend test suite (mock mode, no API key needed)"
	@echo "  make showcase - Synthesize Khafre showcase meshes + manifest"
	@echo "  make clean    - Remove build artifacts and temporary caches"

install:
	pip install -r requirements.txt

api:
	uvicorn apps.api.main:app --reload --port 8000 --host 0.0.0.0

web:
	cd apps/web && npm install && npm run dev

test:
	pytest apps/api/tests -q

showcase:
	python -m apps.api.pipeline.khafre_sar_volume

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
