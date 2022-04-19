# -*- coding: utf-8 -*-


import pickle
import utils
import json

class Jsonl():
    def read(self, input_path):
        """
        Read a jsonl file. Each line is treated a json instance. 
        :param input_path: path tot the input file.
        """
        data = []
        with open(input_path,"r", encoding="utf-8") as file_obj: 
            for json_line in file_obj:
                json_line = json.loads(json_line) # json_line is a dictionary now
                data.append(json_line)            
        return data
    
    def write(self, insts, indices, file_path):
        """
        Write lines in jsonl file based on the given indices
        :param insts: list of all instances
        :param indices: selected indices to store from insts
        :param file_path: path to the output file
        """
        with open(file_path, "w", encoding="utf-8") as file_obj:
            for indx in indices:
                json.dump(insts[indx], file_obj)
                file_obj.write("\n")

def save_pickle_file(info_dict, file_path):
    """
    Save data into a pikle file.
    :param info_dict: Dictionary containing information
    :param file_path: path to the output file
    """
    with open(file_path, "wb") as file_obj:
        pickle.dump(info_dict, file_obj)        
    print("Data created at {}".format(file_path))
        
def read_pickle_file(file_path):
    """
    Read pickle data into a file.
    :param info_dict: Dictionary containing information
    :param file_path: path to the input file        
    """
    with open(file_path, "rb") as file_obj:
        info_dict = pickle.load(file_obj) 
    return info_dict