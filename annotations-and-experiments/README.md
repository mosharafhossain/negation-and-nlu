# Experiments-with-RoBERTa

## Requirements
Python 3.7 
### Install jiant python package 
We utilize the jiant python toolkit. More information on the toolkit can be found [here](https://github.com/nyu-mll/jiant)

```bash
git clone https://github.com/nyu-mll/jiant.git
cd jiant
pip install --upgrade pip
pip install -e .
cd ..
```

## How to Download the Corpora and RoBERTa model
```bash
  python download_model.py
```

## How to Fine-tune

- To fine-tune RoBERTa model with the training dataset of a corpus (e.g., qqp, qnli): 
```bash
  python train.py -c ./config/qqp/config.json
```
  + Arguments:
	  - -c, --config_path: path to the configuration file, (required)
	
## How to Evaluate on the Dev dataset
	
```bash
  python evaluate.py -c ./config/qqp/config.json
```
    + Arguments:
	  - -c, --config_path: path to the configuration file, (required)
	  