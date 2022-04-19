# -*- coding: utf-8 -*-


import os
import numpy as np

class annotation():
                
    def qqp(self, neg_info, insts, dir_path, count=1000):        
        """
        Generate annotation file for qqp. 
        :param neg_info: list of dictionaries containing negation information
                         keys for each dictionary: idx, is_neg, is_neg_text_a, is_neg_text_b, neg_text_a, neg_text_b
        :param insts: list of all instances/lines of the original jsonl file.
        :param file_path: path to the output directory
        """
        
        file_path = os.path.join(dir_path, "qqp_annotation_tab.tsv")
                        
        # get indices of the negated sentences
        w_neg_indices = []
        for index, sent_dict in enumerate(neg_info):
            if sent_dict["is_neg"]:
                w_neg_indices.append(index)                            
                
                
        size_neg = len(w_neg_indices)
        print("Number of Neg instances: {}".format(size_neg))
        selected_indices = np.random.choice(w_neg_indices, min(count, size_neg), replace=False)
        
        
        # write the sentences into a tab generated file
        delim = "\t"
        index = 1
        with open(file_path, "w", encoding="utf-8") as file_obj:
            
            file_obj.write("index" + delim + "dev_index" + delim + "question1" +delim + "cues" + delim + "question2" + delim + "cues" + delim + "judgment" + delim + "is_important")
            file_obj.write("\n")
            for indx in selected_indices:
                                
                dev_index = insts[indx]["idx"]
                question1 = insts[indx]["text_a"].strip('" ')
                cues1     = " | ".join(["{} {}".format(cue,loc) for cue, loc in neg_info[indx]["neg_text_a"] ])
                question2 = insts[indx]["text_b"].strip('" ')
                cues2     = " | ".join(["{} {}".format(cue,loc) for cue, loc in neg_info[indx]["neg_text_b"] ])
                judgment  = insts[indx]["label"]
                is_important = ""
                
                file_obj.write(str(index) + delim + str(dev_index) + delim + question1 +delim + cues1 + delim + question2 + delim + cues2 + delim + judgment + delim + is_important)
                file_obj.write("\n")
                index += 1
                                
    def qnli(self, neg_info, insts, dir_path, count=1000):        
        """
        Generate annotation file for qnli. 
        :param neg_info: list of dictionaries containing negation information. 
                         keys for each dictionary: idx, is_neg, is_neg_text_a, is_neg_text_b, neg_text_a, neg_text_b
        :param insts: list of all instances/lines of the original jsonl file.
        :param file_path: path to the output directory
        """
        
        file_path = os.path.join(dir_path, "qnli_annotation_tab.tsv")
                        
        # get indices of the negated sentences
        w_neg_indices = []
        for index, sent_dict in enumerate(neg_info):
            if sent_dict["is_neg"]:
                w_neg_indices.append(index)                            
                
                            
        size_neg = len(w_neg_indices)
        print("Number of Neg instances: {}".format(size_neg))
        selected_indices = np.random.choice(w_neg_indices, min(count, size_neg), replace=False)
        
        
        # write the sentences into a tab generated file
        delim = "\t"
        index = 1
        with open(file_path, "w", encoding="utf-8") as file_obj:
            
            file_obj.write("index" + delim + "dev_index" + delim + "premise" +delim + "cues" + delim + "hypothesis" + delim + "cues" + delim + "judgment" + delim + "is_important")
            file_obj.write("\n")
            for indx in selected_indices:
                                
                dev_index = insts[indx]["idx"]
                premise = insts[indx]["premise"].strip('" ')  # question
                cues1     = " | ".join(["{} {}".format(cue,loc) for cue, loc in neg_info[indx]["neg_text_a"] ])
                hypothesis = insts[indx]["hypothesis"].strip('" ') #text segment 
                cues2     = " | ".join(["{} {}".format(cue,loc) for cue, loc in neg_info[indx]["neg_text_b"] ])
                judgment  = insts[indx]["label"]
                is_important = ""
                
                file_obj.write(str(index) + delim + str(dev_index) + delim + premise +delim + cues1 + delim + hypothesis + delim + cues2 + delim + judgment + delim + is_important)
                file_obj.write("\n")
                index += 1
                              
                                
    def stsb(self, neg_info, insts, dir_path, count=1000):        
        """
        Generate annotation file for stsb. 
        :param neg_info: list of dictionaries containing negation information. 
                         keys for each dictionary: idx, is_neg, is_neg_text_a, is_neg_text_b, neg_text_a, neg_text_b
        :param insts: list of all instances/lines of the original jsonl file.
        :param file_path: path to the output directory
        """
        
        file_path = os.path.join(dir_path, "stsb_annotation_tab.tsv")
                        
        # get indices of the negated sentences
        w_neg_indices = []
        for index, sent_dict in enumerate(neg_info):
            if sent_dict["is_neg"]:
                w_neg_indices.append(index)                            
                
                            
        size_neg = len(w_neg_indices)
        print("Number of Neg instances: {}".format(size_neg))
        selected_indices = np.random.choice(w_neg_indices, min(count, size_neg), replace=False)
        
        
        # write the sentences into a tab generated file
        delim = "\t"
        index = 1
        with open(file_path, "w", encoding="utf-8") as file_obj:
            
            file_obj.write("index" + delim + "dev_index" + delim + "text_a" +delim + "cues" + delim + "text_b" + delim + "cues" + delim + "judgment" + delim + "is_important")
            file_obj.write("\n")
            for indx in selected_indices:
                                
                dev_index = insts[indx]["idx"]
                premise = insts[indx]["text_a"].strip('" ')  # question
                cues1     = " | ".join(["{} {}".format(cue,loc) for cue, loc in neg_info[indx]["neg_text_a"] ])
                hypothesis = insts[indx]["text_b"].strip('" ') #text segment 
                cues2     = " | ".join(["{} {}".format(cue,loc) for cue, loc in neg_info[indx]["neg_text_b"] ])
                judgment  = str(insts[indx]["label"])
                is_important = ""
                
                file_obj.write(str(index) + delim + str(dev_index) + delim + premise +delim + cues1 + delim + hypothesis + delim + cues2 + delim + judgment + delim + is_important)
                file_obj.write("\n")
                index += 1                   
                
                
    def wic(self, neg_info, insts, dir_path, count=1000):        
        """
        Generate annotation file for wic. 
        :param neg_info: list of dictionaries containing negation information. 
                         keys for each dictionary: idx, is_neg, is_neg_text_a, is_neg_text_b, neg_text_a, neg_text_b
        :param insts: list of all instances/lines of the original jsonl file.
        :param file_path: path to the output directory
        """
        
        file_path = os.path.join(dir_path, "wic_annotation_tab.tsv")
                        
        # get indices of the negated sentences
        w_neg_indices = []
        for index, sent_dict in enumerate(neg_info):
            if sent_dict["is_neg"]:
                w_neg_indices.append(index)                            
                
                            
        size_neg = len(w_neg_indices)
        print("Number of Neg instances: {}".format(size_neg))
        selected_indices = np.random.choice(w_neg_indices, min(count, size_neg), replace=False)
        
        
        # write the sentences into a tab generated file
        delim = "\t"
        index = 1
        with open(file_path, "w", encoding="utf-8") as file_obj:
            
            file_obj.write("index" + delim + "dev_index" + delim + "sentence1" +delim + "cues" + delim + "sentence2" + delim + "cues" + delim +"target_word"+ delim+ "judgment" + delim + "is_important")
            file_obj.write("\n")
            for indx in selected_indices:
                                
                dev_index = insts[indx]["idx"]
                premise = insts[indx]["sentence1"].strip('" ')  # question
                cues1     = " | ".join(["{} {}".format(cue,loc) for cue, loc in neg_info[indx]["neg_text_a"] ])
                hypothesis = insts[indx]["sentence2"].strip('" ') #text segment 
                cues2     = " | ".join(["{} {}".format(cue,loc) for cue, loc in neg_info[indx]["neg_text_b"] ])
                target_word = insts[indx]["word"]
                judgment  = str(insts[indx]["label"])
                is_important = ""
                
                file_obj.write(str(index) + delim + str(dev_index) + delim + premise +delim + cues1 + delim + hypothesis + delim + cues2 + delim + target_word + delim + judgment + delim + is_important)
                file_obj.write("\n")
                index += 1
                
                                    
    def commonsenseqa(self, neg_info, insts, dir_path, count=1000):        
        """
        Generate annotation file for commonsenseqa. 
        :param neg_info: list of dictionaries containing negation information. 
                         keys for each dictionary: idx, is_neg, is_neg_text_a, is_neg_text_b, neg_text_a, neg_text_b
        :param insts: list of all instances/lines of the original jsonl file.
        :param file_path: path to the output directory
        """
        
        file_path = os.path.join(dir_path, "commonsenseqa_annotation_tab.tsv") #change
                        
        # get indices of the negated sentences
        w_neg_indices = []
        for index, sent_dict in enumerate(neg_info):
            if sent_dict["is_neg_text_a"]:
                w_neg_indices.append(index)                            
                
                            
        size_neg = len(w_neg_indices)
        print("Number of Neg instances: {}".format(size_neg))
        selected_indices = np.random.choice(w_neg_indices, min(count, size_neg), replace=False)
        
        
        # write the sentences into a tab generated file
        delim = "\t"
        index = 1
        with open(file_path, "w", encoding="utf-8") as file_obj:
            
            file_obj.write("index" + delim + "dev_index" + delim + "question" +delim + "cues" + delim + "choices" + delim + "judgment" + delim + "is_important")  #change
            file_obj.write("\n")
            for indx in selected_indices:
                                
                dev_index = "NA"
                question = insts[indx]["question"].strip('" ')  #change
                cues1     = " | ".join(["{} {}".format(cue,loc) for cue, loc in neg_info[indx]["neg_text_a"] ])
                choices   = str(insts[indx]["choices"])
                judgment  = str(insts[indx]["answerKey"])
                is_important = ""
                
                file_obj.write(str(index) + delim + str(dev_index) + delim + question +delim + cues1 + delim + choices + delim + judgment + delim + is_important)
                file_obj.write("\n")
                index += 1
                
    
    def sst(self, neg_info, insts, dir_path, count=1000):        
        """
        Generate annotation file for SST-2. 
        :param neg_info: list of dictionaries containing negation information. 
                         keys for each dictionary: idx, is_neg, is_neg_text_a, is_neg_text_b, neg_text_a, neg_text_b
        :param insts: list of all instances/lines of the original jsonl file.
        :param file_path: path to the output directory
        """
        
        file_path = os.path.join(dir_path, "sst_annotation_tab.tsv") #change
                        
        # get indices of the negated sentences
        w_neg_indices = []
        for index, sent_dict in enumerate(neg_info):
            if sent_dict["is_neg_text_a"]:
                w_neg_indices.append(index)                            
                
                            
        size_neg = len(w_neg_indices)
        print("Number of Neg instances: {}".format(size_neg))
        selected_indices = np.random.choice(w_neg_indices, min(count, size_neg), replace=False)
        
        
        # write the sentences into a tab generated file
        delim = "\t"
        index = 1
        with open(file_path, "w", encoding="utf-8") as file_obj:
            
            file_obj.write("index" + delim + "dev_index" + delim + "text" +delim + "cues" + delim + "judgment" + delim + "is_important")  #change
            file_obj.write("\n")
            for indx in selected_indices:
                                
                dev_index = insts[indx]["idx"]
                text = insts[indx]["text"].strip('" ')  #change
                cues1     = " | ".join(["{} {}".format(cue,loc) for cue, loc in neg_info[indx]["neg_text_a"] ])
                judgment  = str(insts[indx]["label"])
                is_important = ""
                
                file_obj.write(str(index) + delim + str(dev_index) + delim + text +delim + cues1 + delim + judgment + delim + is_important)
                file_obj.write("\n")
                index += 1
                
    def wsc(self, neg_info, insts, dir_path, count=1000):        
        """
        Generate annotation file for wsc. 
        :param neg_info: list of dictionaries containing negation information. 
                         keys for each dictionary: idx, is_neg, is_neg_text_a, is_neg_text_b, neg_text_a, neg_text_b
        :param insts: list of all instances/lines of the original jsonl file.
        :param file_path: path to the output directory
        """
        
        file_path = os.path.join(dir_path, "wsc_annotation_tab.tsv") #change
                        
        # get indices of the negated sentences
        w_neg_indices = []
        for index, sent_dict in enumerate(neg_info):
            if sent_dict["is_neg_text_a"]:
                w_neg_indices.append(index)                            
                
                            
        size_neg = len(w_neg_indices)
        print("Number of Neg instances: {}".format(size_neg))
        selected_indices = np.random.choice(w_neg_indices, min(count, size_neg), replace=False)
        
        
        # write the sentences into a tab generated file
        delim = "\t"
        index = 1
        with open(file_path, "w", encoding="utf-8") as file_obj:
            
            file_obj.write("index" + delim + "dev_index" + delim + "text" +delim + "cues" + delim + "target" + delim + "judgment" + delim + "is_important")  #change
            file_obj.write("\n")
            for indx in selected_indices:
                                
                dev_index = insts[indx]["idx"]
                text = insts[indx]["text"].strip('" ')  #change
                cues1     = " | ".join(["{} {}".format(cue,loc) for cue, loc in neg_info[indx]["neg_text_a"] ])
                target    = str(insts[indx]["target"])
                judgment  = str(insts[indx]["label"])
                is_important = ""
                
                file_obj.write(str(index) + delim + str(dev_index) + delim + text +delim + cues1 + delim + target + delim + judgment + delim + is_important)
                file_obj.write("\n")
                index += 1
        
