setup:
	python3 -m venv ~/.company_bankruptcy_predictions

install:
	pip install --upgrade pip &&\
          pip install -r requirements.txt

test:
	python3 -m pytest --nbval Notebook_bankrupt.ipynb
  
all: install test
