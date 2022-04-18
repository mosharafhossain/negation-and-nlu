# -*- coding: utf-8 -*-

import torch
import os
from preprocessing import Dataprep, Connl_format
from collections import defaultdict 
def generate_vocabs(datasets, unknown):
    """
    Generate vocabulary for the model.
    """
    
    vocabs = {"cues":{"PAD":0, "B_C":1, "I_C":2, "O_C":3 } }
    """
    vocabs = {}    
    for f, value in unknown.items():
        all_f = []
        for key in datasets.keys():
            all_f += datasets[key]
        unique = list(np.unique(all_f))
    """ 
        
    return vocabs


def safe_division(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0.0

def save_in_file(file_path, tuple_list):
    """

    :param file_path:
    :param tuple_list:
    :return:
    """
    tuple_list = sorted(tuple_list, key=lambda x:x[2], reverse=True)
    delim = "\t"
    with open(file_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("gold_cue" + delim + "predicted_cue" + delim + "count")
        file_obj.write("\n")
        for gold_cue, pred_cue, count in tuple_list:
            file_obj.write(gold_cue + delim + pred_cue + delim + str(count))
            file_obj.write("\n")

def save_cues_info(cue_dict, dir_path):
    """

    :param cue_dict:
    :param dir_path:
    :return:
    """

    true_positives = []  # model detect them as cues and they are actually cues
    false_positives = []  # model detects them as cues although they are not actually cues
    false_negatives = []  # model don't detect them as cues although they are actually cues
    for gold_cue, pred_cue_dict in cue_dict.items():
        #sorted_tuples = sorted(pred_cue_dict.items(), key=lambda x: x[1], reverse=True)
        #print("gold_cue: {}, sorted_tuples: {}".format(gold_cue, sorted_tuples))
        for pred_cue, count in pred_cue_dict.items():
            if gold_cue.strip().lower() == pred_cue.strip().lower():
                true_positives.append((gold_cue, pred_cue, count))
            elif gold_cue.strip().lower() == "not_a_cue":
                false_positives.append((gold_cue, pred_cue, count))
            elif pred_cue.strip().lower() == "not_detected":
                false_negatives.append((gold_cue, pred_cue, count))
            else:
                assert 1==2

    true_positives_path = os.path.join(dir_path, "true_positives.txt")
    false_positives_path = os.path.join(dir_path, "false_positives.txt")
    false_negatives_path = os.path.join(dir_path, "false_negatives.txt")
    save_in_file(true_positives_path, true_positives)
    save_in_file(false_positives_path, false_positives)
    save_in_file(false_negatives_path, false_negatives)
    

def save_sentences_with_prediction(dev_data, pred_positions, file_path):
    """
    This method saves the sentences including gold and predicted cues.
    :param dev_data (dictionary): contains all the information of a dataset.
    :param pred_positions (list of list): position of predicted cues.Example for a sentence: [[2,5], [10]] means, first cue is located in the index 2 and 5, and, the second cue is located at position 10.
    :param file_path: path to the output file.
    """
    gold_positions = dev_data["cue_positions"]
    sent_len = len(gold_positions)
    delim = "\t"
    
    assert len(gold_positions) == len(pred_positions)
    
    with open(file_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("gold_cue" + delim + "predicted_cue" + delim + "sentence")
        file_obj.write("\n")
        
        for i in range(sent_len):
            gold_pos = gold_positions[i]
            pred_pos = pred_positions[i]
            sent     = dev_data["tokens"][i]
            sent_text = " ".join(sent)
            sent_pos = list(range(len(sent)))
            
            #print("gold_pos: {}, pred_pos: {}".format(gold_pos, pred_pos))
            #print("sent_pos: {}".format(sent_pos))
            
            gold_pos_text = ""
            pred_pos_text = ""
            
            for pos_seq in gold_pos:  #e.g. pos_seq= [[2,5], [10]]
                gold_pos_text += "["
                for pos in pos_seq:
                    if pos in sent_pos:
                        gold_pos_text += sent[pos]
                gold_pos_text += "] "
                    
                    
            for pos_seq in pred_pos:  #e.g. pos_seq= [[2,5], [10]]
                pred_pos_text += "["
                for pos in pos_seq:
                    if pos in sent_pos:
                        pred_pos_text += sent[pos]
                pred_pos_text += "] "                    
                        
                    
            if len(gold_pos_text)==0:
                gold_pos_text = "[]"
            if len(pred_pos_text)==0:
                pred_pos_text = "[]"
                    
            file_obj.write(gold_pos_text + delim + pred_pos_text + delim + sent_text)
            file_obj.write("\n")
                
        
                
        
def measures_cue(gold_labels, pred_output, gold_positions, vocabs):
    """

    :param gold_labels:
    :param pred_output:
    :param gold_positions:
    :param vocabs:
    :return:
    """
    cues_itos = {i:s for s,i in vocabs["cues"].items()}
    pred_output = [[cues_itos[ind] for ind in seq] for seq in pred_output]
    gold_labels = [[cues_itos[ind] for ind in seq] for seq in gold_labels]

    #print("gold_labels[0:x]: {}".format(gold_labels[0:5]))
    #print("pred_output[0:x]: {}".format(pred_output[0:5]))
    
    matched   = 0
    predicted = 0
    gold      = 0
    for i in range(len(gold_positions)):
        gold_pos = gold_positions[i]
        pred_pos = Dataprep().get_cue_positions(pred_output[i]) 
        predicted += sum([1 if len(e)>=1 else 0 for e in pred_pos]) #len(pred_pos)
        gold      += sum([1 if len(e)>=1 else 0 for e in gold_pos]) #len(gold_pos)
        #print("gold_pos: {}, pred_pos: {}".format(gold_pos, pred_pos))
        for j in range(len(pred_pos)):            
            if pred_pos[j] in gold_pos: 
                matched += 1

    precision = safe_division(matched, predicted)
    recall    = safe_division(matched, gold)
    f1_score  = safe_division(2*precision*recall, precision+recall)
    measures  = {"matched":matched, "predicted": predicted, "gold":gold, 
                 "precision": round(precision*100, 2), "recall": round(recall*100, 2), "f1_score": round(f1_score*100, 2)}
    return measures


def predict_only(pred_output, dev_data, vocabs, output_path):
    """

    :param gold_labels:
    :param pred_output:
    :param gold_positions:
    :param vocabs:
    :return:
    """
    cues_itos = {i:s for s,i in vocabs["cues"].items()}
    pred_output = [[cues_itos[ind] for ind in seq] for seq in pred_output]

    predicted_cues = 0
    pred_positions = []
    num_sent_cues = 0
    num_sents = len(pred_output)
    for i in range(num_sents):
        pred_pos = Dataprep().get_cue_positions(pred_output[i]) 
        pred_positions.append(pred_pos)
        sent_cues = sum([1 if len(e)>=1 else 0 for e in pred_pos]) #len(pred_pos)
        predicted_cues += sent_cues
        if sent_cues > 0:
            num_sent_cues += 1
            
    print("num_sents: {}, num_sents_with_cues: {}, all_cue_count = {}".format(num_sents, num_sent_cues, predicted_cues))

    if output_path:
        #write into file
        delim = "\t"
        with open(output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("negation_cues" + delim + "cue_positions" + delim + "sentence_tokens" )
            file_obj.write("\n")
            
            for i in range(num_sents):
                pred_pos = pred_positions[i]
                sent     = dev_data["tokens"][i]
                sent_text = " ".join(sent)
                sent_pos = list(range(len(sent)))
     
                pred_pos_text = ""         
                for pos_seq in pred_pos:  #e.g. pos_seq= [[2,5], [10]]
                    #pred_pos_text += "["
                    for pos in pos_seq:
                        if pos in sent_pos:
                            pred_pos_text += sent[pos] + " "
                    pred_pos_text += "| "                    
    
                if len(pred_pos_text)==0:
                    pred_pos_text = "no-cues"
                        
                pred_pos = "[" + " | ".join([ ",".join([str(e) for e in l]) for l in  pred_pos]) + "]"
                file_obj.write(pred_pos_text.strip(" |") + delim + pred_pos + delim + sent_text)
                file_obj.write("\n")

    


def measures_cue_cdsco(pred_output, dev_data, vocabs):
    """
    Get evaluation results best on cd-sco evaluation script.
    :param pred_output:
    :param gold_positions:
    :param vocabs:
    :return:
    """
    cue_dict = defaultdict(lambda: defaultdict(int))
    gold_positions = dev_data["cue_positions"]
    cues_itos = {i: s for s, i in vocabs["cues"].items()}
    pred_output = [[cues_itos[ind] for ind in seq] for seq in pred_output]

    matched = 0
    predicted = 0
    gold = 0
    pred_positions = []
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(gold_positions)):
        gold_pos = gold_positions[i]
        pred_pos = Dataprep().get_cue_positions(pred_output[i])
        pred_positions.append(pred_pos)
        predicted += sum([1 if len(e) >= 1 else 0 for e in pred_pos])  # len(pred_pos)
        gold += sum([1 if len(e) >= 1 else 0 for e in gold_pos])  # len(gold_pos)
        # print("gold_pos: {}, pred_pos: {}".format(gold_pos, pred_pos))
        for j in range(len(pred_pos)):
            if pred_pos[j] in gold_pos:
                matched += 1

        tokens = dev_data["tokens"][i]
        for j in range(len(gold_pos)):
            if gold_pos[j] in pred_pos:  # True positive cases
                tp += 1
            else:  # False Negative Cases
                fn += 1

        # False Positive Cases
        for j in range(len(pred_pos)):
            if pred_pos[j] not in gold_pos:
                if sum([1 if len( set(pred_pos[j]).intersection(set(gold_pos[k])) ) > 0 else 0 for k in range(len(gold_pos))]) > 0:
                    fn += 1
                else:
                    fp += 1

    precision = safe_division(tp, (tp+fp))
    recall = safe_division(tp, (tp+fn))
    f1_score = safe_division(2 * precision * recall, precision + recall)
    measures = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
                "f1_score": f1_score}
    return measures


def measures_cue_save(pred_output, dev_data, vocabs, params, dataset_key, obj_list):
    """

    :param pred_output:
    :param dev_data:
    :param vocabs:
    :param params:
    :return:
    """
    cue_dict = defaultdict(lambda: defaultdict(int))
    gold_positions = dev_data["cue_positions"]
    cues_itos = {i:s for s,i in vocabs["cues"].items()}
    pred_output = [[cues_itos[ind] for ind in seq] for seq in pred_output]

    
    matched   = 0
    predicted = 0
    gold      = 0
    pred_positions = []
    for i in range(len(gold_positions)):
        gold_pos = gold_positions[i]
        pred_pos = Dataprep().get_cue_positions(pred_output[i])
        pred_positions.append(pred_pos)
        predicted += sum([1 if len(e)>=1 else 0 for e in pred_pos]) #len(pred_pos)
        gold      += sum([1 if len(e)>=1 else 0 for e in gold_pos]) #len(gold_pos)
        #print("gold_pos: {}, pred_pos: {}".format(gold_pos, pred_pos))
        for j in range(len(pred_pos)):            
            if pred_pos[j] in gold_pos: 
                matched += 1
        
        tokens = dev_data["tokens"][i]
        for j in range(len(gold_pos)):
            gold_cue = " ".join([tokens[idx] for idx in gold_pos[j] ])
            if gold_pos[j] in pred_pos: # True positive cases
                pidx = pred_pos.index(gold_pos[j])
                pred_cue = " ".join([tokens[idx] for idx in pred_pos[pidx] ])
                cue_dict[gold_cue.lower()][pred_cue.lower()] += 1
            else: # False Negative Cases
                pred_cue = "not_detected"
            if len(gold_cue.strip()) > 0: # considering cases where gold cues are not empty
                cue_dict[gold_cue.lower()][pred_cue.lower()] += 1

        # False Positive Cases
        for j in range(len(pred_pos)):
            pred_cue = " ".join([tokens[idx] for idx in pred_pos[j] ])
            if pred_pos[j] not in gold_pos: 
                cue_dict["not_a_cue"][pred_cue.lower()] += 1

    precision = safe_division(matched, predicted)
    recall    = safe_division(matched, gold)
    f1_score  = safe_division(2*precision*recall, precision+recall)
    measures  = {"matched":matched, "predicted": predicted, "gold":gold, "precision":precision, "recall":recall, "f1_score":f1_score}


    # Save the Cues
    dir_path = params["output_dirs"][dataset_key]
    save_cues_info(cue_dict, dir_path)
    pred_path = os.path.join(dir_path, "prediction.txt")
    connl = Connl_format()
    connl.prepare_connl_file(obj_list, pred_positions, pred_path)
    
    # Save the sentence file
    pred_path = os.path.join(dir_path, "prediction_plaintext.txt")
    save_sentences_with_prediction(dev_data, pred_positions, pred_path)
    
    return measures
                
