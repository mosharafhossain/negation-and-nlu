# -*- coding: utf-8 -*-


import sys
sys.path.insert(0, "./jiant")

import os

import jiant.utils.python.io as py_io
import jiant.scripts.download_data.runscript as downloader

import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.utils.display as display

import argparse
import json 


argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config_path", help="path of the configuration file", required=True)      
args        = argParser.parse_args()
config_path = args.config_path


# Read parameters from json file
with open(config_path) as json_file_obj: 
	params = json.load(json_file_obj)

 

TASK_NAME = params["task_name"]
MODEL_TYPE = params["model_type"]
os.makedirs("./run_configs/", exist_ok=True)

# Download model
if params["is_download_model"]:
    export_model.lookup_and_export_model(
        model_type=MODEL_TYPE,
        output_base_path="./models/" + MODEL_TYPE,
    )
    

# Tokenize and cache-----------------------------------------------------
tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
    task_config_path=f"./content/exp/tasks/configs/{TASK_NAME}_config.json",
    model_type=MODEL_TYPE,
    model_tokenizer_path=params["model_tokenizer_path"],
    output_dir=f"./outputs/{TASK_NAME}",
    phases=["train", "val"],
))
print("Tokenization is completed!")

# Write a run config -----------------------------------------------------------
jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
    task_config_base_path="./content/exp/tasks/configs",
    task_cache_base_path="./outputs",
    train_task_name_list=[TASK_NAME],
    val_task_name_list=[TASK_NAME],
    train_batch_size= params["train_batch_size"],
    eval_batch_size= params["eval_batch_size"], 
    epochs= params["num_epochs"],
    num_gpus= params["num_gpus"],
    #warmup_steps_proportion=params["warmup_steps_proportion"]
).create_config()
py_io.write_json(jiant_run_config, params["run_config"]) 
display.show_json(jiant_run_config)
print("Configuration is set up!")


# Start training ------------------------------------------------------------------------
print("Training Started----------")
run_args = main_runscript.RunConfiguration(
    jiant_task_container_config_path=params["run_config"],
    output_dir="./runs/"+TASK_NAME,
    model_type=MODEL_TYPE,
    model_path=params["model_path"],
    model_config_path=params["model_config_path"],
    model_tokenizer_path=params["model_tokenizer_path"],
    learning_rate= params["learning_rate"],
    eval_every_steps=50000,
    do_train=True,
    do_val=True,
    do_save=True,
    force_overwrite=True,
    seed=params["seed"]
)
main_runscript.run_loop(run_args)
print("Training is completed!")