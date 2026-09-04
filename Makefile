.PHONY: help install clean train submit all

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make train      - Train all models"
	@echo "  make submit     - Generate submission files"
	@echo "  make clean      - Clean temporary files"
	@echo "  make all        - Run full pipeline"

install:
	pip install -r requirements.txt

train:
	python main.py

submit:
	python main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf models/*.pkl models/*.h5 models/*.cbm
	rm -rf submissions/*.csv
	rm -rf outputs/*.png outputs/*.json

all: clean install train submit
