# negation-and-nlu
This repository will contain the code and data of the ACL2022 paper "An Analysis of Negation in Natural Language Understanding Corpora".


## Requirements
Python 3.6+  
#### Create virtual environment and install packages
```bash
# Create virtual environment -> optional step
python3 -m venv your_location/negation-and-nlu
source your_location/negation-and-nlu/bin/activate

# Install required packages
pip install --upgrade pip
pip install torch torchvision
pip install -U spacy
python -m spacy download en_core_web_sm
pip install transformers
pip install git+https://github.com/kmkurn/pytorch-crf
```

#### Download RoBERTa-base 
RoBERTa-base model can be downloaded from [here](https://huggingface.co/roberta-base) and need to stored in the directory "./model/pre-trained/roberta_base".
Below lines of command will download the necessary files to the target directory.

```bash
cd ./model/pre-trained/roberta_base
wget https://huggingface.co/roberta-base/resolve/main/config.json
wget https://huggingface.co/roberta-base/resolve/main/merges.txt
wget https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin
wget https://huggingface.co/roberta-base/resolve/main/tokenizer.json
wget https://huggingface.co/roberta-base/resolve/main/vocab.json
cd ../../../
```


## How to Fine-tune

- To fine-tune RoBERTa model on CD-SCO corpus run the below command: 
```bash
  python train.py -c ./config/train.json 
```
  + Arguments:
	  - -c, --config_path: path to the configuration file, (required)


## How to Evaluate on New Data
Evaluate the test split of CD-SCO corpus:  

```bash
  python evaluate.py --config_path ./config/predict.json
```

Detect negation cues in a text file:  
Each line in the file should contain a single sentence.

```bash
  python evaluate.py --config_path ./config/predict.json --input_path ./outputs/sample_io/input_file.txt --output_path ./outputs/sample_io/output_file.txt
```
  + Arguments:  
	  - --config-path: path to the configuration file; (required). It contains all hyperparameter settings.  
	  - --input_path: path to the input file (one sentence per line)  
	  - --output_path: path to the output file (each line in the output file contains the list of cues, cue positions, and the original sentence (tab separated)). 
 	  
	  


