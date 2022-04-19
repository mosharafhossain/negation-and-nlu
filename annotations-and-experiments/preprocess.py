# -*- coding: utf-8 -*-

import json
import argparse

class ReadJsonl():
    def read(self, input_path, sent1_header=None, sent2_header=None):
        """
        Read a jsonl file. Each line is treated a json instance. 
        :param input_path: path tot the input file.
        :param sent1_header: header name for sentence 1
        :param sent2_header: header name for sentence 2
        """
        data = {}
        sent1_list = []
        sent2_list = []
        with open(input_path,"r", encoding="utf-8") as file_obj: 
            for json_line in file_obj:
                json_line = json.loads(json_line) # json_line is a dictionary now
                if sent1_header:
                    sent1 = json_line[sent1_header].strip()
                    if len(sent1) > 0:
                        sent1_list.append(sent1)
                    else:
                        sent1_list.append("[BLANK SPACE]")
                if sent2_header:
                    sent2 = json_line[sent2_header].strip()
                    if len(sent2) > 0:
                        sent2_list.append(sent2)
                    else:
                        sent2_list.append("[BLANK SPACE]")
            
            # Quality check
            if sent2_header:
                assert len(sent1_list) == len(sent2_list)

                
        data["sent1"] = sent1_list
        if sent2_header:
            data["sent2"] = sent2_list
            
        return data
    
    def write(self, data, file_list):
        """
        Write sentences into files. Each output file contains single sentence per line.
        :param data (dict): If the input file contans two sentences(e.g., premise, hypothesis) in each example, then keys are: sent1, sent2
        :param file_list: list of all output files. 
        """
        keys = data.keys()
        
        assert len(keys) == len(file_list)        
        for i in range(len(file_list)):
            if i == 0:
                key = "sent1"  # First file contains the sentences appeared first in the input file
            elif i == 1:
                key = "sent2"  # Second file contains the sentences appeared second in the input file
            
            with open(file_list[i], "w", encoding="utf-8") as file_obj:
                for sent in data[key]:
                    file_obj.write(sent)
                    file_obj.write("\n")
                    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="path to the input file", required=True)
    parser.add_argument("--sent1_h", help="header name of sentence 1", required=True)
    parser.add_argument("--sent2_h", help="header name of sentence 2", required=False, default=None)
    parser.add_argument("--out_path1", help="path to the first output file", required=True)
    parser.add_argument("--out_path2", help="path to the second output file", required=False, default=None)
    
    args = parser.parse_args()
    input_path    = args.input_path
    sent1_h       = args.sent1_h
    sent2_h       = args.sent2_h
    output_path1  = args.out_path1
    output_path2  = args.out_path2
    output_list = [output_path1, output_path2]
    
    
    jsonl = ReadJsonl()
    data  = jsonl.read(input_path, sent1_h, sent2_h)
    print("Data Read completed!")
    file_list = [file_path for file_path in output_list if file_path]
    jsonl.write(data, file_list)
    print("Data Write completed!")
    
    
    
    
    
    
    