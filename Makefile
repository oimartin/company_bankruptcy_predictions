install:
	pip install --upgrade pip &&\
          pip install -r requirements.txt

test:
	python -m pytest --nbval Notebook_bankrupt.ipynb
  
all: install test
