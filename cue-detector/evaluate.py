# -*- coding: utf-8 -*-

import torch
import numpy as np
import random
import os
import argparse
import json 
import tqdm
from transformers import (BertTokenizer, 
                          AlbertTokenizer, 
                          #T5Tokenizer,
                          RobertaTokenizer,
                          AutoTokenizer,
                          BertConfig, 
                          AdamW,
                          get_linear_schedule_with_warmup)

from module.model import DetectNeg
from module.batch import Batchprep
from preprocessing import Dataprep, DataprepFile
import util



def set_seed(seed=42):
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)	
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # required for using multi-GPUs.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


# Command line arguments
# python evaluate.py --config_path ./config/config.json
argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config_path", help="path of the configuration file", required=True) 
argParser.add_argument("--input_path", help="path to the input file", required=False, default=None) 
argParser.add_argument("--output_path", help="path to the output file", required=False, default=None)      
args        = argParser.parse_args()
config_path = args.config_path
input_path  = args.input_path
output_path = args.output_path



# Read parameters from json file
with open(config_path) as json_file_obj: 
	params = json.load(json_file_obj)


# Set the seed    
set_seed(params["seed"]) 
use_fewshot = params["use_fewshot"]
if use_fewshot:
    lan = params["fewshot_lan"]
    model_path = params["fewshot_shot_model"][lan]
    eval_paths = [(lan, params["test_path_fewshot"][lan])]
else:
    model_path = params["best_model_path"]
    eval_paths = [ (name, path) for name, path in params["evaluate_paths"].items()]
    
# Model Load
map_location = 'cuda:{}'.format(params["device"]) if params["use_gpu"] else 'cpu'
state = torch.load(model_path, map_location=map_location)

vocabs = state['vocabs']
model = DetectNeg(params, vocabs)
model.load_state_dict(state['model'])

if torch.cuda.is_available()==True: 
    device = torch.device("cuda:"+str(params["device"]))
else: 
    device = torch.device("cpu") 

model.to(device) 
model.eval()  

# Get the tokenizer
if params["pt_system"] == "mBERT-base":
    tokenizer = BertTokenizer.from_pretrained(params["mBERT-base-path"], do_lower_case=False)
elif params["pt_system"] == "XLM-RoBERTa-base":
    tokenizer = AutoTokenizer.from_pretrained(params["XLM-RoBERTa-base-path"])
elif params["pt_system"] == "albert":
    tokenizer = AlbertTokenizer.from_pretrained(params["albert-path"], do_lower_case=False)
elif params["pt_system"] == "RoBERTa-base":
    tokenizer = RobertaTokenizer.from_pretrained(params["RoBERTa-base-path"], do_lower_case=False)

if input_path:
    dev_data = DataprepFile().preprocess(input_path)
    dev_size = len(dev_data["tokens"])
    
    dev_output = []
    dev_batch_num = int(np.ceil(dev_size/params["dev_batch_size"]))
    dev_iterator = Batchprep().get_a_batch(params, vocabs, tokenizer, dev_data, dev_size, dev_batch_num, device, shuffle=False)
    for dev_batch_idx in range(dev_batch_num):
        dev_batch_data, dev_batch_labels = next(dev_iterator)
        dev_batch_output = model.predict(dev_batch_data)
        
        dev_output.extend(dev_batch_output)   
        
    measures = util.predict_only(dev_output, dev_data, vocabs, output_path)

else:
    for key, file_name in eval_paths:  #example "english_test":"./data/cd-sco/corpus/test/test.txt"
        print("Started Evaluating for {}. File location: {}".format(key, file_name))

        dev_data, obj_list = Dataprep().preprocess(file_name, phase_name="Dev", isIncludeNonCue=True)
        dev_size = len(dev_data["tokens"])
        
        dev_output = []
        dev_gold   = []
        dev_batch_num = int(np.ceil(dev_size/params["dev_batch_size"]))
        dev_iterator = Batchprep().get_a_batch(params, vocabs, tokenizer, dev_data, dev_size, dev_batch_num, device, shuffle=False)
        for dev_batch_idx in range(dev_batch_num):
            dev_batch_data, dev_batch_labels = next(dev_iterator)
            dev_batch_output = model.predict(dev_batch_data)
            
            dev_output.extend(dev_batch_output)   
            dev_gold.extend(dev_batch_labels.tolist())
            
        #measures = util.measures_cue(dev_gold, dev_output, dev_data["cue_positions"], vocabs)
        #measures = util.measures_cue_cdsco(dev_output, dev_data, vocabs)
        measures = util.measures_cue_save(dev_output, dev_data, vocabs, params, key, obj_list)
        print("Current best Measures: {}".format(measures))
        print("Ended Evaluation for {}\n\n".format(os.path.basename(file_name)))



