# -*- coding: utf-8 -*-

import pickle
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

from preprocessing import Dataprep
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



argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config_path", help="path of the configuration file", required=True)      
args        = argParser.parse_args()
config_path = args.config_path


# Read parameters from json file
with open(config_path) as json_file_obj: 
	params = json.load(json_file_obj)


# Set the seed    
set_seed(params["seed"]) 


# Data preparation__________________________________________________________________
train_data, _ = Dataprep().preprocess(params["train_path"], phase_name="Training", isIncludeNonCue=True)
dev_data, _ = Dataprep().preprocess(params["dev_path"], phase_name="Development", isIncludeNonCue=True)
vocabs  = util.generate_vocabs(train_data, unknown={"cues":False, "upos":True})
train_size = len(train_data["tokens"])
dev_size = len(dev_data["tokens"])

# Save vocab
with open(params["vocab_dir"], "wb") as file_obj:
	pickle.dump(vocabs, file_obj)

# Get the tokenizer__________________________________________________________________________
if params["pt_system"] == "mBERT-base":
    tokenizer = BertTokenizer.from_pretrained(params["mBERT-base-path"], do_lower_case=False)
elif params["pt_system"] == "XLM-RoBERTa-base":
    tokenizer = AutoTokenizer.from_pretrained(params["XLM-RoBERTa-base-path"])
#elif params["pt_system"] == "mT5-base":
#    tokenizer = T5Tokenizer.from_pretrained(params["mT5-base-path"])    
elif params["pt_system"] == "albert":
    tokenizer = AlbertTokenizer.from_pretrained(params["albert-path"], do_lower_case=False)
elif params["pt_system"] == "RoBERTa-base":
    tokenizer = RobertaTokenizer.from_pretrained(params["RoBERTa-base-path"], do_lower_case=False)
    

if torch.cuda.is_available()==True and params["use_gpu"]: 
    device = torch.device("cuda:"+str(params["device"]))
else: 
    device = torch.device("cpu") 

model = DetectNeg(params, vocabs)
if params["use_multi_gpus"]:
    model = torch.nn.DataParallel(model)
    model.to(device)  
    state = dict(model=model.module.state_dict(), vocabs=vocabs)
else:
    model.to(device)  
    state = dict(model=model.state_dict(), vocabs=vocabs)


# optimizer
param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
        'lr': params["bert_learning_rate"], 'weight_decay': params["bert_weight_decay"]
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith('bert') and 'crf' not in n],
        'lr': params["learning_rate"], 'weight_decay': params["weight_decay"]
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith('bert') and 'crf' in n],
        'lr': params["learning_rate"], 'weight_decay': 0
    }
]


batch_num = int(np.ceil(train_size/params["batch_size"]))
#optimizer = AdamW(model.parameters())
optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num * params["warmup_epoch"],
                                           num_training_steps=batch_num * params["max_epoch"])


eval_ = True
best_f1 = -1.0
for epoch in range(params["max_epoch"]):
    # training set
    progress = tqdm.tqdm(total=batch_num, ncols=75, desc='Train Epoch {}/{}'.format(epoch+1, params["max_epoch"] ))
    optimizer.zero_grad()
    
    data_iterator = Batchprep().get_a_batch(params, vocabs, tokenizer, train_data, train_size, batch_num, device, shuffle=True)    
    #print("Test 6") 
    for batch_idx in range(batch_num):
        batch_data, batch_labels = next(data_iterator)
        
        loss = model(batch_data, batch_labels)  
        loss = loss * (1 / params["accumulate_step"])
        if params["use_multi_gpus"]:
            loss.mean().backward()
        else:
            loss.backward()
        

        if (batch_idx + 1) % params["accumulate_step"] == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), params["grad_clipping"])
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            
    if eval_:  
        dev_output = []
        dev_gold   = []
        dev_batch_num = int(np.ceil(dev_size/params["dev_batch_size"]))
        dev_iterator = Batchprep().get_a_batch(params, vocabs, tokenizer, dev_data, dev_size, dev_batch_num, device, shuffle=False)
        for dev_batch_idx in range(dev_batch_num):
            dev_batch_data, dev_batch_labels = next(dev_iterator)
            if params["use_multi_gpus"]:
                dev_batch_output = model.module.predict(dev_batch_data)
            else:
                dev_batch_output = model.predict(dev_batch_data)
            dev_output.extend(dev_batch_output)   
            dev_gold.extend(dev_batch_labels.tolist())
        measures = util.measures_cue(dev_gold, dev_output, dev_data["cue_positions"], vocabs)        
        if measures["f1_score"] > best_f1:
            best_f1 = measures["f1_score"]
            print('\nSaving the best model at {}'.format(params["best_model_path"]))
            torch.save(state, params["best_model_path"])
            print("Current best Measures: {}\n\n".format(measures))
            patience = 0
        else:
            patience += 1
            if patience > params["patience"]:
                break
            
    progress.close()  
    
    
    


