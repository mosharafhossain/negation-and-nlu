# -*- coding: utf-8 -*-


import json
import os
import utils
from annotate import annotation

import numpy as np
np.random.seed(44)
from collections import Counter
import argparse

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
    
    

def read_pred_file(file_path, is_header=True):
    """
    Read a file containing prediction of negation cues. The file contains three headers (negation_cues, cue_positions, sentence_tokens)
    :param file_path: path to the input file
    :param is_header: whether the file has a header
    """
    insts = []
    with open(file_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if is_header:
                is_header= False
                continue
            else:
                tokens = line.strip().split("\t")
                if len(line.strip()) != 0:
                    insts.append(tokens)
    return insts



class dataset2Texts():
    def read_neg(self, file_text_a, file_text_b, is_header=True):
        """
        Read files and extract negation information.
        :param file_text_a: path to the file for text_a
        :param file_text_b: path to the file for text_b
        :param is_header: whether the file has a header
        """
        
        insts_text_a = read_pred_file(file_text_a, is_header)
        insts_text_b = read_pred_file(file_text_b, is_header)
        #print("insts_text_a: {}, {}".format(len(insts_text_a), insts_text_a[-3:len(insts_text_a)]))
        #print("insts_text_b: {}, {}".format(len(insts_text_b), insts_text_b[-3:len(insts_text_b)]))
        assert len(insts_text_a) == len(insts_text_b)
        
        results = []
        size = len(insts_text_a)
        
        for i in range(size):
            inst_a =  insts_text_a[i]
            inst_b =  insts_text_b[i]
            
            is_neg_text_a = False
            is_neg_text_b = False
            neg_text_a    = []
            neg_text_b    = []
            
            if inst_a[0].strip() == "no-cues" and inst_b[0].strip() == "no-cues":   #index 0 contains cues in plain text or "no-cues"
                result = {"idx": i, 
                          "is_neg": False, 
                          "is_neg_text_a": is_neg_text_a,
                          "is_neg_text_b": is_neg_text_b,
                          "neg_text_a": neg_text_a,
                          "neg_text_b":neg_text_b
                          }            
            else:
                            
                if inst_a[0].strip() != "no-cues":
                    is_neg_text_a = True
                    neg_cues = [cue.strip() for cue in inst_a[0].strip().split("|")]  #index 0 contains negation cues
                    neg_loc  = [loc.strip() for loc in  inst_a[1].strip("[] ").split("|")] #index 1 contains positions of the negation cues
                    assert len(neg_cues) == len(neg_loc)
                    
                    for cue, loc in zip(neg_cues, neg_loc): #neg_loc can be ["2,6" "10"] for example  neg_cues = ["neither nor", "not"]
                        #print("cue: {}, pos: {}".format(cue, [pos.strip() for pos in loc.split(",")]))
                        loc_list = [int(pos.strip()) for pos in loc.split(",")]
                        neg_text_a.append( (cue, loc_list) )
    
                if inst_b[0].strip() != "no-cues":
                    is_neg_text_b = True
                    neg_cues = [cue.strip() for cue in inst_b[0].strip().split("|")]
                    neg_loc  = [loc.strip() for loc in  inst_b[1].strip("[] ").split("|")]
                    assert len(neg_cues) == len(neg_loc)
                    
                    for cue, loc in zip(neg_cues, neg_loc): #neg_loc can be ["2,6" "10"] for example  neg_cues = ["neither nor", "not"]
                        #print("cue: {}, pos: {}".format(cue, [pos.strip() for pos in loc.split(",")]))
                        loc_list = [int(pos.strip()) for pos in loc.split(",")]
                        neg_text_b.append( (cue, loc_list) )
                    
                result = {"idx": i, 
                          "is_neg": True, 
                          "is_neg_text_a": is_neg_text_a,
                          "is_neg_text_b": is_neg_text_b,
                          "neg_text_a": neg_text_a,
                          "neg_text_b": neg_text_b
                          }
            
            results.append(result)
            
        return results
    
    def read_neg_onesentence(self, file_text_a, is_header=True):
        """
        Read files and extract negation information for the tasks that contain one sentence for each instance.
        :param file_text_a: path to the file for text_a
        :param is_header: whether the file has a header
        """
        
        insts_text_a = read_pred_file(file_text_a, is_header)        
        
        results = []
        size = len(insts_text_a)
        
        for i in range(size):
            inst_a =  insts_text_a[i]
            #print("inst_a: {}".format(inst_a))
            
            is_neg_text_a = False
            neg_text_a    = []
            
            if inst_a[0].strip() == "no-cues":   #index 0 contains cues in plain text or "no-cues"
                result = {"idx": i, 
                          "is_neg": False, 
                          "is_neg_text_a": is_neg_text_a,
                          "neg_text_a": neg_text_a
                          }            
            else:
                is_neg_text_a = True
                neg_cues = [cue.strip() for cue in inst_a[0].strip().split("|")]  #index 0 contains negation cues
                neg_loc  = [loc.strip() for loc in  inst_a[1].strip("[] ").split("|")] #index 1 contains positions of the negation cues
                assert len(neg_cues) == len(neg_loc)
                
                for cue, loc in zip(neg_cues, neg_loc): #neg_loc can be ["2,6" "10"] for example  neg_cues = ["neither nor", "not"]
                    #print("cue: {}, pos: {}".format(cue, [pos.strip() for pos in loc.split(",")]))
                    loc_list = [int(pos.strip()) for pos in loc.split(",")]
                    neg_text_a.append( (cue, loc_list) )
    
                result = {"idx": i, 
                          "is_neg": True, 
                          "is_neg_text_a": is_neg_text_a,
                          "neg_text_a": neg_text_a
                          }
            
            results.append(result)
            
        return results
    
    def extract_save(self, neg_info, insts, dir_path):        
        """
        Extract the indices of with and without neg sentences and save the files.
        :param neg_info: list of dictionaries containing negation information. 
                         keys for each dictionary: idx, is_neg, is_neg_text_a, is_neg_text_b, neg_text_a, neg_text_b
        :param insts: list of all instances of the original jsonl file
        :param dir_path: path to the output directory
        """
        
        
        wo_neg       = []
        
        w_neg_any    = []
        w_neg_sent_a = []
        w_neg_sent_b = []
        w_neg_both   = []
        
        for index, sent_dict in enumerate(neg_info):
            if not sent_dict["is_neg"]:
                wo_neg.append(index)
            else:
                w_neg_any.append(index)                
                if sent_dict["is_neg_text_a"] and sent_dict["is_neg_text_b"]:
                    w_neg_both.append(index)
                elif sent_dict["is_neg_text_a"]:
                    w_neg_sent_a.append(index)
                elif sent_dict["is_neg_text_b"]:
                    w_neg_sent_b.append(index)
                    
        
        print("w_neg_sent_a: {}".format(len(w_neg_sent_a)))
        print("w_neg_sent_b: {}".format(len(w_neg_sent_b)))
        print("w_neg_both: {}".format(len(w_neg_both)))
        print("w_neg_any: {}, all: {}".format(len(w_neg_any), len(w_neg_sent_a)+ len(w_neg_sent_b)+ len(w_neg_both) ))
        print("wo_neg: {}".format(len(wo_neg)))
        
        #write the files
        Jsonl().write(insts, w_neg_sent_a, os.path.join(dir_path, "w_neg_sent_a.jsonl"))
        Jsonl().write(insts, w_neg_sent_b, os.path.join(dir_path, "w_neg_sent_b.jsonl"))
        Jsonl().write(insts, w_neg_both, os.path.join(dir_path, "w_neg_both.jsonl"))
        Jsonl().write(insts, w_neg_any, os.path.join(dir_path, "w_neg_any.jsonl"))
        Jsonl().write(insts, wo_neg, os.path.join(dir_path, "wo_neg.jsonl"))
        
        #write 
        utils.save_pickle_file(neg_info, os.path.join(dir_path, "dev_neg_info.pkl"))
        
    
    def extract_save_onesentence(self, neg_info, insts, dir_path):        
        """
        Extract the indices of with and without neg sentences for the tasks that contain only one sentence per instance.
        :param neg_info: list of dictionaries containing negation information. 
                         keys for each dictionary: idx, is_neg, is_neg_text_a, is_neg_text_b, neg_text_a, neg_text_b
        :param insts: list of all instances of the original jsonl file
        :param dir_path: path to the output directory
        """
        
        
        wo_neg       = []
        w_neg_sent_a = []
        
        for index, sent_dict in enumerate(neg_info):
            if not sent_dict["is_neg_text_a"]:
                wo_neg.append(index)
            else:
                w_neg_sent_a.append(index)                
                                    
        
        print("w_neg_sent_a: {}".format(len(w_neg_sent_a)))        
        print("wo_neg: {}".format(len(wo_neg)))
        
        #write the filesJsonl().write(insts, w_neg_sent_a, os.path.join(dir_path, "w_neg_sent_a.jsonl"))
        Jsonl().write(insts, wo_neg, os.path.join(dir_path, "wo_neg.jsonl"))
        
        
        #write 
        utils.save_pickle_file(neg_info, os.path.join(dir_path, "dev_neg_info.pkl"))
                
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", help="name of the corpus", required=True)
    parser.add_argument("--annotations", help="Number of annotations to write", required=True)
    
    args        = parser.parse_args()
    corpus_name = args.corpus
    num_annotations = int(args.annotations)
    
    if corpus_name == "qqp":
        # read val split
        file_path = "./content/exp/tasks/data/qqp/val.jsonl"
        insts = Jsonl().read(file_path)
        print("# lines: {}".format(len(insts)))
        
        
        #read neg files
        file_text_a = "./content/exp/tasks/data/qqp/negation/predicted_sentences/val_text_a.txt"
        file_text_b = "./content/exp/tasks/data/qqp/negation/predicted_sentences/val_text_b.txt"
        results = dataset2Texts().read_neg(file_text_a, file_text_b) #list of dictionaries
        
        
        # save into files
        dir_path = "./content/exp/tasks/data/qqp/negation/eval_splits"
        
        #prepare annotation file
        annotation().qqp(results, insts, dir_path, count=num_annotations)
        
        
    if corpus_name == "qnli":
        # read val split
        file_path = "./content/exp/tasks/data/qnli/val.jsonl"
        insts = Jsonl().read(file_path)
        print("# lines: {}".format(len(insts)))
        #print("Lines: {}".format(insts[0:5]))
        
        
        #read neg files
        file_text_a = "./content/exp/tasks/data/qnli/negation/predicted_sentences/val_premise.txt"   #premise is the question here
        file_text_b = "./content/exp/tasks/data/qnli/negation/predicted_sentences/val_hypothesis.txt"  #hypothesis is the text segment here
        results = dataset2Texts().read_neg(file_text_a, file_text_b) #list of dictionaries
        
        print("\n# results: {}".format(len(results)))
        #print("results: {}".format(["{}".format(r) for r in results[-40:40430] if r["is_neg"]]))
        
        
        # Split corpora
        dir_path = "./content/exp/tasks/data/qnli/negation/eval_splits"
        dataset2Texts().extract_save(results, insts, dir_path)
        
        #prepare annotation file
        annotation().qnli(results, insts, dir_path, count=num_annotations)
         
        
    if corpus_name == "stsb":
        # read val split
        file_path = "./content/exp/tasks/data/"+corpus_name+"/val.jsonl"
        insts = Jsonl().read(file_path)
        print("# lines: {}".format(len(insts)))
        #print("Lines: {}".format(insts[0:5]))
                
        #read neg files
        file_text_a = "./content/exp/tasks/data/"+corpus_name+"/negation/predicted_sentences/val_text_a.txt"   
        file_text_b = "./content/exp/tasks/data/"+corpus_name+"/negation/predicted_sentences/val_text_b.txt"  
        results = dataset2Texts().read_neg(file_text_a, file_text_b) #list of dictionaries
        
        print("\n# results: {}".format(len(results)))
        #print("results: {}".format(["{}".format(r) for r in results[-40:40430] if r["is_neg"]]))
        
        
        # Split corpora
        dir_path = "./content/exp/tasks/data/"+corpus_name+"/negation/eval_splits"
        dataset2Texts().extract_save(results, insts, dir_path)
        
        #prepare annotation file
        annotation().stsb(results, insts, dir_path, count=num_annotations)
        
        
    if corpus_name == "wic":
        # read val split
        file_path = "./content/exp/tasks/data/"+corpus_name+"/val.jsonl"
        insts = Jsonl().read(file_path)
        print("# lines: {}".format(len(insts)))
        #print("Lines: {}".format(insts[0:5]))
                
        #read neg files
        file_text_a = "./content/exp/tasks/data/"+corpus_name+"/negation/predicted_sentences/val_sentence1.txt"   
        file_text_b = "./content/exp/tasks/data/"+corpus_name+"/negation/predicted_sentences/val_sentence2.txt"  
        results = dataset2Texts().read_neg(file_text_a, file_text_b) #list of dictionaries
        
        print("\n# results: {}".format(len(results)))
        #print("results: {}".format(["{}".format(r) for r in results[-40:40430] if r["is_neg"]]))
        
        
        # Split corpora
        dir_path = "./content/exp/tasks/data/"+corpus_name+"/negation/eval_splits"
        dataset2Texts().extract_save(results, insts, dir_path)
        
        #prepare annotation file
        annotation().wic(results, insts, dir_path, count=num_annotations)
        

        
    if corpus_name == "commonsenseqa":
        # read val split
        file_path = "./content/exp/tasks/data/"+corpus_name+"/val.jsonl"
        insts = Jsonl().read(file_path)
        print("# lines: {}".format(len(insts)))
        #print("Lines: {}".format(insts[0:5]))
                
        #read neg files
        file_text_a = "./content/exp/tasks/data/"+corpus_name+"/negation/predicted_sentences/val_question.txt"     #change        
        results = dataset2Texts().read_neg_onesentence(file_text_a) #list of dictionaries
        
        print("\n# instances: {}".format(len(results)))
        #print("results: {}".format(["{}".format(r) for r in results[0:20] if r["is_neg"]]))
        
        
        # Split corpora
        dir_path = "./content/exp/tasks/data/"+corpus_name+"/negation/eval_splits"
        dataset2Texts().extract_save_onesentence(results, insts, dir_path)
        
        #prepare annotation file
        annotation().commonsenseqa(results, insts, dir_path, count=num_annotations) #change
        
        
    if corpus_name == "sst":
        # read val split
        file_path = "./content/exp/tasks/data/"+corpus_name+"/val.jsonl"
        insts = Jsonl().read(file_path)
        print("# lines: {}".format(len(insts)))
        #print("Lines: {}".format(insts[0:5]))
                
        #read neg files
        file_text_a = "./content/exp/tasks/data/"+corpus_name+"/negation/predicted_sentences/val_text.txt"     #change        
        results = dataset2Texts().read_neg_onesentence(file_text_a) #list of dictionaries
        
        print("\n# instances: {}".format(len(results)))
        #print("results: {}".format(["{}".format(r) for r in results[0:20] if r["is_neg"]]))
        
        
        # Split corpora
        dir_path = "./content/exp/tasks/data/"+corpus_name+"/negation/eval_splits"
        dataset2Texts().extract_save_onesentence(results, insts, dir_path)
        
        #prepare annotation file
        annotation().sst(results, insts, dir_path, count=num_annotations) #change
        

    if corpus_name == "wsc":
        # read val split
        file_path = "./content/exp/tasks/data/"+corpus_name+"/val.jsonl"
        insts = Jsonl().read(file_path)
        print("# lines: {}".format(len(insts)))
        #print("Lines: {}".format(insts[0:5]))
                
        #read neg files
        file_text_a = "./content/exp/tasks/data/"+corpus_name+"/negation/predicted_sentences/val_text.txt"     #change        
        results = dataset2Texts().read_neg_onesentence(file_text_a) #list of dictionaries
        
        print("\n# instances: {}".format(len(results)))
        #print("results: {}".format(["{}".format(r) for r in results[0:20] if r["is_neg"]]))
        
        
        # Split corpora
        dir_path = "./content/exp/tasks/data/"+corpus_name+"/negation/eval_splits"
        dataset2Texts().extract_save_onesentence(results, insts, dir_path)
        
        #prepare annotation file
        annotation().wsc(results, insts, dir_path, count=num_annotations) #change
        
        
        
        