# -*- coding: utf-8 -*-

from utils import Jsonl
import argparse
import os

def read_file_twosent(file_path, num_attributes=8, is_header=True):
    """
    Read a annotation file that has two sentences per instance.
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
                if num_attributes == 9:
                    is_important_idx = 8
                else:
                    is_important_idx = 7
                #print(tokens)
                if len(tokens) == num_attributes:
                    info_dict = {"index": int(tokens[0].strip()),
                                 "dev_index": int(tokens[1].strip()),
                                 "sent1": tokens[2].strip(),
                                 "cues1": tokens[3].strip(),
                                 "sent2": tokens[4].strip(),
                                 "cues2": tokens[5].strip(),
                                 "judgment": tokens[6].strip(),
                                 "is_important": tokens[is_important_idx].strip(),
                        }

                    insts.append(info_dict)
    return insts


def read_file_onesent(file_path, num_attributes=6, is_header=True):
    """
    Read a annotation file that has two sentences per instance.
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
                if num_attributes == 7:
                    is_important_idx = 6
                    judgement_idx    = 5
                else:
                    is_important_idx = 5
                    judgement_idx    = 4
                    
                
                #print(tokens)
                if len(tokens) == num_attributes:
                    info_dict = {"index": int(tokens[0].strip()),
                                 "dev_index": tokens[1].strip(),
                                 "sent1": tokens[2].strip(),
                                 "cues1": tokens[3].strip(),
                                 "judgment": tokens[judgement_idx].strip(),
                                 "is_important": tokens[is_important_idx].strip(),
                        }

                    insts.append(info_dict)
    return insts

def extract_important(insts):
    """
    Extract important/unimportant pairs.
    :param insts: list of dictionry containing information.
    """
    imp_insts = []
    unimp_ints = []
    num_annotations = 0
    for sent_json in insts:
        if sent_json["is_important"].lower() == "yes":
            imp_insts.append(sent_json)
            num_annotations += 1
        elif sent_json["is_important"].lower() == "no":
            unimp_ints.append(sent_json)
            num_annotations += 1
    print("all instances: {}".format(len(insts)))
    print("#annotated instances: {}, imp: {}, unimp: {}".format(num_annotations, len(imp_insts), len(unimp_ints) ))
    return imp_insts, unimp_ints

def map_and_save(insts, imp_insts, unimp_ints, dir_path):
    """
    Map important/unimportant instances with the original instances. Then save into two files.
    :param insts: list of dictionries containing information.
    :param imp_insts: list of dictionries containing information of important negation type.
    :param unimp_ints: list of dictionries containing information of unimportant negation type.
    :param dir_path: path to the output directory
    """
    
    imp_indices = [int(sent_json["dev_index"]) for sent_json in imp_insts]
    unimp_indices = [int(sent_json["dev_index"]) for sent_json in unimp_ints]
    
    # get indices from val split.
    imp_indices_orig = []
    unimp_indices_orig = []
    for idx, json_line in enumerate(insts):
        if int(insts[idx]["idx"]) in imp_indices:
            imp_indices_orig.append(idx)
        elif int(insts[idx]["idx"]) in unimp_indices:
            unimp_indices_orig.append(idx)
    
    annotatated_indices = imp_indices_orig + unimp_indices_orig
    assert len(annotatated_indices) == len(imp_indices) + len(unimp_indices)
    
    # Write to the file
    print("all: {}, imp: {}, unimp: {}".format(len(annotatated_indices), len(imp_indices_orig), len(unimp_indices_orig) ))
    Jsonl().write(insts, annotatated_indices, os.path.join(dir_path, "w_neg_annotated.jsonl"))
    Jsonl().write(insts, imp_indices_orig, os.path.join(dir_path, "w_neg_important.jsonl"))
    Jsonl().write(insts, unimp_indices_orig, os.path.join(dir_path, "w_neg_unimportant.jsonl"))
    

def map_and_save_v2(insts, imp_insts, unimp_ints, dir_path, sent_name):
    """
    Map important/unimportant instances with the original instances. Then save into two files.
    :param insts: list of dictionries containing information.
    :param imp_insts: list of dictionries containing information of important negation type.
    :param unimp_ints: list of dictionries containing information of unimportant negation type.
    :param dir_path: path to the output directory
    :param sent_name: name of the attribute that holds sentence of each instance
    """
    
    imp_sentences = [sent_json["sent1"] for sent_json in imp_insts]
    unimp_sentences = [sent_json["sent1"] for sent_json in unimp_ints]
    
    # get indices from val split.
    imp_indices_orig = []
    unimp_indices_orig = []
    for idx, json_line in enumerate(insts):
        if insts[idx][sent_name] in imp_sentences:
            imp_indices_orig.append(idx)
        elif insts[idx][sent_name] in unimp_sentences:
            unimp_indices_orig.append(idx)
    
    annotatated_indices = imp_indices_orig + unimp_indices_orig
    assert len(annotatated_indices) == len(imp_sentences) + len(unimp_sentences)
    
    # Write to the file
    print("all: {}, imp: {}, unimp: {}".format(len(annotatated_indices), len(imp_indices_orig), len(unimp_indices_orig) ))
    Jsonl().write(insts, annotatated_indices, os.path.join(dir_path, "w_neg_annotated.jsonl"))
    Jsonl().write(insts, imp_indices_orig, os.path.join(dir_path, "w_neg_important.jsonl"))
    Jsonl().write(insts, unimp_indices_orig, os.path.join(dir_path, "w_neg_unimportant.jsonl"))


if __name__ == "__main__":
    
    #python post_annotation.py --corpus qqp
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", help="name of the corpus", required=True)
    
    args        = parser.parse_args()
    corpus_name = args.corpus
    file_name = corpus_name+".tsv"
    file_path = os.path.join( "./annotations/", file_name)
    dir_path = "./content/exp/tasks/data/"+corpus_name+"/negation/eval_splits"
    
    if corpus_name in ["qqp", "qnli", "stsb", "wic"]:   
        
        # Extract important/unimportant splits
        num_attributes = 9 if corpus_name == "wic" else 8
        anno_insts = read_file_twosent(file_path, num_attributes)        
        imp_insts, unimp_ints = extract_important(anno_insts)
        
        # Read original dev split
        file_path = "./content/exp/tasks/data/"+corpus_name+"/val.jsonl"
        insts = Jsonl().read(file_path)
        
        # Save files
        map_and_save(insts, imp_insts, unimp_ints, dir_path)
        
    if corpus_name in ["commonsenseqa"]:   
        # Extract important/unimportant splits
        num_attributes = 7
        anno_insts = read_file_onesent(file_path, num_attributes)        
        imp_insts, unimp_ints = extract_important(anno_insts)
        
        # Read original dev split
        file_path = "./content/exp/tasks/data/"+corpus_name+"/val.jsonl"
        insts = Jsonl().read(file_path)
        
        # Save files
        map_and_save_v2(insts, imp_insts, unimp_ints, dir_path, "question")
        
    if corpus_name in ["sst"]:   
        # Extract important/unimportant splits
        num_attributes = 6
        anno_insts = read_file_onesent(file_path, num_attributes)        
        imp_insts, unimp_ints = extract_important(anno_insts)
        print("YES")
        # Read original dev split
        file_path = "./content/exp/tasks/data/"+corpus_name+"/val.jsonl"
        insts = Jsonl().read(file_path)
        
        # Save files
        map_and_save(insts, imp_insts, unimp_ints, dir_path)
        
        
    if corpus_name in ["wsc"]:   
        # Extract important/unimportant splits
        num_attributes = 7
        anno_insts = read_file_onesent(file_path, num_attributes)        
        imp_insts, unimp_ints = extract_important(anno_insts)
        
        # Read original dev split
        file_path = "./content/exp/tasks/data/"+corpus_name+"/val.jsonl"
        insts = Jsonl().read(file_path)
        
        # Save files
        map_and_save(insts, imp_insts, unimp_ints, dir_path)
        
        
        
        
        
        
        
        