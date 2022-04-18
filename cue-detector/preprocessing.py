
from collections import defaultdict
import spacy
import numpy as np
nlp = spacy.load("en_core_web_sm")
import re
np.random.seed(44)

class data_structure:
    def __init__(self, line, delim="\t"):
        """
        Create data_structure instance for each line of the input data file.
        :param line (str): line of the input file
        """
        if "\n" in line:
            line = line[0:-1].strip()
        tokens = line.split(delim)
        self.chap_name = tokens[0]
        self.sent_num = tokens[1]
        self.token_num = tokens[2]
        self.word = tokens[3]
        self.lemma = tokens[4]
        self.pos = tokens[5]
        self.syntax = tokens[6]
        self.negation_list = []
        #print("line: {}".format(tokens))
        #print("tokens[7]: {}".format(tokens[7:]))

        if tokens[7].strip() != "***":
            neg_tokens = tokens[7:]
            l = len(neg_tokens)
            i = 0
            while i < l:
                t = (neg_tokens[i], neg_tokens[ i +1], neg_tokens[ i +2])
                self.negation_list.append(t)
                i = i+ 3


class Dataprep():
    def file_read(self, input_file):
        """
        Read a cd-sco data file.
        :param input_file (str): path to the input file
        :return:
        """
        list_objs = []
        with open(input_file, "r") as file_obj:
            for line in file_obj:
                if len(line.strip()) >= 10: # check the line has values
                    obj = data_structure(line)
                    list_objs.append(obj)
        return list_objs

    def get_syntactical_info(self, sentence, tool="spacy"):
        """
        Extract syntactical information.
        :param sentence: untokenized sentence.
        :return upos, synt_dep: upos is universal parts-of-speech, synt_dep is syntactic dependancy tags
        """
        if tool == "spacy":
            sentence = " ".join(sentence)  # creating a sentence from a list of words
            sentence = nlp(sentence)
            upos = [token.pos_ for token in sentence]
            synt_dep = [token.dep_ for token in sentence]
        return upos, synt_dep

    # Prepares a dictionary of data includes all availavle and made-of features.
    def get_data_details(self, list_objs):
        """
        Extract information from each line in a data file.
        :param list_objs (list): each instance contains data_structure object
        :return data_dict (dict): dictionary of data. Each key is a unique key of chapter name and sentence number.
        """
        data_dict = {}
        word_list = []
        lemma_list = []
        pos_list = []
        cues_dict = defaultdict(list)  # Store cue information for a sentence
        scope_dict = defaultdict(list)  # Store Scope information
        event_dict = defaultdict(list)  # Store Negated event information
        begin_check = defaultdict(lambda: defaultdict(bool))

        sent_num = 0
        num_words = 0
        for i in range(len(list_objs)):

            # Extract features information
            unique_tuple = (list_objs[i].chap_name, list_objs[i].sent_num)
            word_list.append(list_objs[i].word)
            lemma_list.append(list_objs[i].lemma)
            pos_list.append(list_objs[i].pos)
            # Extract Negation information
            neg_list = list_objs[i].negation_list
            num_cues = len(neg_list)
            num_words += 1

            if num_cues > 0:
                for j in range(num_cues):
                    if neg_list[j][0] != "_":
                        if begin_check["cue"][j] == False:
                            cues_dict[j].append("B_C")      # B_C indicates Begin-Cue
                            begin_check["cue"][j] = True
                        else:
                            cues_dict[j].append("I_C")     # I_C indicates Inside-Cue
                    else:
                        cues_dict[j].append("O_C")         # O_C indicates Outside-Cue

                    if neg_list[j][1] != "_":
                        if begin_check["scope"][j] == False:
                            scope_dict[j].append("B_S")
                            begin_check["scope"][j] = True
                        else:
                            scope_dict[j].append("I_S")
                    else:
                        scope_dict[j].append("O_S")

                    if neg_list[j][2] != "_":
                        if begin_check["event"][j] == False:
                            event_dict[j].append("B_E")
                            begin_check["event"][j] = True
                        else:
                            event_dict[j].append("I_E")
                    else:
                        event_dict[j].append("O_E")

                    # Check for reaching last token of a sentence
            if (i == len(list_objs) - 1) or (i + 1 < len(list_objs) and int(list_objs[i + 1].token_num) == 0):
                #num_words = int(list_objs[i].token_num) + 1   # last token_num + 1 gives total token numbers
                upos_list, synt_dep_list = self.get_syntactical_info(word_list)
                data_dict[unique_tuple] = [num_words, num_cues, word_list, lemma_list, pos_list, upos_list, synt_dep_list, cues_dict, scope_dict, event_dict]
                
                #if unique_tuple[0] == "lavadoras_yes_5_7" and unique_tuple[1] == "39":
                #    print("unique_tuple: {}, data_dict[unique_tuple]: {}".format(unique_tuple, data_dict[unique_tuple]))
                # Reset all variables
                word_list = []
                lemma_list = []
                pos_list = []
                cues_dict = defaultdict(list)
                scope_dict = defaultdict(list)
                event_dict = defaultdict(list)
                begin_check = defaultdict(lambda: defaultdict(bool))
                sent_num += 1
                num_words = 0
                #print("Sentence: {}".format(sent_num))

        return data_dict

    def process_data(self, detail_data_dict, isIncludeNonCue=False):
        """
        Process each sentence. Each feature of a sentence is stored in a list. If a sentence contains multiple negation (say, 2 cues), then
        there will be multiple (say, 2) separate instances for that sentence. The separate sentences for that single sentence are
        identified by the same unique key (chapter_name, sentence_number).
        :param detail_data_dict (dict): Data dictionary containing information of each sentence.
                                        Dictionary key is (chapter_name, sentence_number)
        :param isIncludeNonCue: whether includes sentences without negation cue.
        :return data: dictionary of data. each key is a list. Available keys: tokens,
                                                                              lemma (existing),
                                                                              pos (existing),
                                                                              upos (by spaCy),
                                                                              sdep (by spaCy),
                                                                              cues,
                                                                              scopes,
                                                                              events
        """
        neg_sent_count  = 0
        neg_count       = 0
        data = defaultdict(list)
        for key, value in detail_data_dict.items():
            num_tokens = value[0]  # index 0 stores number of tokens in a sentence
            num_cues = value[1]    # index 1 stores number of cues
            if num_cues > 0:
                neg_sent_count += 1
                for i in range(num_cues):  # does not include the data with no cues
                    data["key"].append(key)
                    data["num_tokens"].append(num_tokens)
                    data["tokens"].append(value[2])     # index 2 stores list of words of a sentence
                    data["lemma"].append(value[3])      # index 3 stores list of lemmas of the words of a sentence
                    data["pos"].append(value[4])        # index 4 stores list of PoSs of a sentence
                    data["upos"].append(value[5])       # index 5 stores list of universal PoSs of a sentence
                    data["sdep"].append(value[6])       # index 6 stores list of syntactic dependency in a sentences
                    data["cues"].append(value[7][i])    # index 7 stores dictionary of cues in a sentences
                    data["scopes"].append(value[8][i])  # index 8 stores dictionary of scopes in a sentences
                    data["events"].append(value[9][i])  # index 9 stores dictionary of negated events in a sentences
                    neg_count += 1
            else:
                if isIncludeNonCue == True:
                    data["key"].append(key)
                    data["num_tokens"].append(num_tokens)
                    data["tokens"].append(value[2])  # index 2 stores list of words of a sentence
                    data["lemma"].append(value[3])  # index 3 stores list of lemmas of the words of a sentence
                    data["pos"].append(value[4])  # index 4 stores list of PoSs of a sentence
                    data["upos"].append(value[5])  # index 5 stores list of universal PoSs of a sentence
                    data["sdep"].append(value[6])  # index 6 stores list of syntactic dependency in a sentences
                    data["cues"].append(["O_C" for _ in range(num_tokens)])
                    data["scopes"].append(["O_S" for _ in range(num_tokens)])
                    data["events"].append(["O_E" for _ in range(num_tokens)])
        print("# Negated sentences: {}, # Negation cues: {}".format(neg_sent_count, neg_count))
        return data
    
    def get_individual_cue_positions(self, seq):
        """
        Get the positions where the tokens of a single cue are located.
        :param seq: tokenized sequence for cue of a sentence. Each token can be either B_C or I_C or O_C.
        """
        positions = []
        size = len(seq)
        for i in range(size):
            if seq[i] in ("B_C", "I_C"):
                positions.append(i)
        return positions
    
    
    def get_cue_positions(self, seq):
        """
        Get the positions where the tokens of a all cues of a sentence are located.
        An example for a sentence: [[2,5], [10]] means, first cue is located in the index 2 and 5, and, the second cue is located at position 10.
        If there is no cue in a sentence then cue_positions = [].
        :param seq: tokenized sequence for cue of a sentence. Each token can be either B_C or I_C or O_C or PAD.
        """
        positions = []
        cur_pos = []
        size = len(seq)
        for i in range(size):
            if seq[i] == "B_C" or i == size-1:
                if cur_pos:
                    positions.append(cur_pos)
                cur_pos = [i]
            elif seq[i] == "I_C":
                cur_pos.append(i)
        return positions
    
    def process_data_joint(self, detail_data_dict, isIncludeNonCue=False):
        """
        prepare token-level sequences. Each feature of a sentence is stored in a list. If a sentence contain multiple cues, then only one list
        is created for cue sequence. Note that scope and event sequences might be problemic to work (work on it).
        :param detail_data_dict (dict): Data dictionary containing information of each sentence.
                                        Dictionary key is (chapter_name, sentence_number)
        :param isIncludeNonCue: whether includes sentences without negation cue.
        :return data: dictionary of data. each key is a list. Available keys: tokens,
                                                                              lemma (existing),
                                                                              pos (existing),
                                                                              upos (by spaCy),
                                                                              sdep (by spaCy),
                                                                              cues,
                                                                              scopes,
                                                                              events
        """
        neg_sent_count  = 0
        neg_count       = 0
        all_sent_count  = 0
        data = defaultdict(list)
        for key, value in detail_data_dict.items():
            num_tokens = value[0]  # index 0 stores number of tokens in a sentence
            num_cues = value[1]    # index 1 stores number of cues            
            if num_cues > 0:
                neg_sent_count += 1
                neg_count += num_cues
                all_sent_count += 1
                data["key"].append(key)
                data["num_tokens"].append(num_tokens)
                data["tokens"].append(value[2])     # index 2 stores list of words of a sentence
                data["lemma"].append(value[3])      # index 3 stores list of lemmas of the words of a sentence
                data["pos"].append(value[4])        # index 4 stores list of PoSs of a sentence
                data["upos"].append(value[5])       # index 5 stores list of universal PoSs of a sentence
                data["sdep"].append(value[6])       # index 6 stores list of syntactic dependency in a sentences                    
                
                cue_seq   = value[7][0]
                scope_seq = value[8][0]
                event_seq = value[9][0]
                
                cue_positions = []
                cue_pos = self.get_individual_cue_positions(cue_seq)
                if cue_pos: cue_positions.append(cue_pos)
                
                for i in range(1, num_cues):  
                    cue_list = value[7][i]   
                    scope_list = value[8][i]   
                    event_list = value[9][i]
                    
                    cue_pos = self.get_individual_cue_positions(cue_list)
                    if cue_pos: cue_positions.append(cue_pos)
                    
                    for j in range(num_tokens):
                        # prepare cue sequence
                        #print("key: {}, num_cues: {}".format(key, num_cues))
                        #print("cue_list: {}, num_tokens: {}, sent: {}\n".format(len(cue_list), num_tokens, len(data["tokens"][i])))
                        if cue_list[j] != "O_C": 
                            cue_seq[j] = cue_list[j]  # Note, two negation cues in a sentence do not overlap
                        
                        # prepare scope sequence
                        if scope_list[j] != "O_S" and scope_seq[j] == "O_S": 
                            scope_seq[j] = scope_list[j]+"_{}".format(i)                          
                        elif scope_list[j] != "O_S" and scope_seq[j] != "O_S": 
                            scope_seq[j] = "C_S" #assign common scope tag
                            
                        # prepare event sequence
                        if event_list[j] != "O_E": 
                            event_seq[j] = event_list[j]  # Note, two negated events in a sentence do not overlap (assumed, please check)
                
                data["cues"].append(cue_seq)
                data["scopes"].append(scope_seq)
                data["events"].append(event_seq)
                data["cue_positions"].append(cue_positions)  
                """
                if num_cues != len(cue_positions):
                    print("num_cues: {}, cue_positions: {}".format(num_cues, cue_positions))
                    print("tokens: {}".format(value[2]))
                    print("cues: {}".format(value[7]))
                    print("cue seq: {}\n".format(cue_seq))
                """
            else:
                if isIncludeNonCue == True:
                    all_sent_count += 1
                    data["key"].append(key)
                    data["num_tokens"].append(num_tokens)
                    data["tokens"].append(value[2])  # index 2 stores list of words of a sentence
                    data["lemma"].append(value[3])  # index 3 stores list of lemmas of the words of a sentence
                    data["pos"].append(value[4])  # index 4 stores list of PoSs of a sentence
                    data["upos"].append(value[5])  # index 5 stores list of universal PoSs of a sentence
                    data["sdep"].append(value[6])  # index 6 stores list of syntactic dependency in a sentences
                    data["cues"].append(["O_C" for _ in range(num_tokens)])
                    data["scopes"].append(["O_S" for _ in range(num_tokens)])
                    data["events"].append(["O_E" for _ in range(num_tokens)])
                    data["cue_positions"].append([])
        print("# sentences: {} | # Negated sentences: {} | # Negation cues: {}".format(all_sent_count, neg_sent_count, neg_count))
        return data
    
    def preprocess(self, file_name, phase_name, isIncludeNonCue=False):
        """
        Preprocess the cd-sco dataset (train or dev or test). 
        :param file_name (str): path to the file.
        :return proc_data (dict): Dictionary containing the data
        """
        #print("Preprocess phase: {}".format(phase_name))
        obj_list  = self.file_read(file_name)
        #print("Step 1: File reading is complete.")
        data_dict = self.get_data_details(obj_list)
        #print("Step 2: Data extraction is complete.")
        proc_data = self.process_data_joint(data_dict, isIncludeNonCue)
        #print("Step 3: Data pre-processing is complete.\n")
        return proc_data, obj_list
        
    
    def quality_check(self, data_dict, num_inst, features):
        """
        Check whether preprocessing is alright or not. 
        :param data_dict (dict): the dictionary that contains the features (each key is a feature) as well as labels.
        :num_inst (int): number of instanct to check
        :features (list): The features we care to check
        """
        
        count = min(num_inst, len(data_dict["key"]) )
        for i in range(count):
            print("i: {}".format(i))
            for f in features:
                print("{}: {}".format(f, data_dict[f][i]))
                
            
            print("  ".join( ["{}/{}/{}/{}".format(data_dict["tokens"][i][ind], data_dict["cues"][i][ind], data_dict["scopes"][i][ind], data_dict["events"][i][ind]) for ind in range(len(data_dict["tokens"][i])) ] ))
            #print(" ".join( ["{}/{}/{}/{}".format(data_dict["tokens"][0][ind], data_dict["cues"][0][ind], data_dict["scopes"][0][ind], data_dict["events"][0][ind]) for ind in range(len(data_dict["tokens"][0])) ] ))
            print("\n")
      

class FileOperations():
    def save(self, list_objs, file_path):
        """
        Save list of "data_structure" instances into a file maintaining CoNNL file format.
        :param list_objs (list): each index contains a "data_structure" class instance.
        :param file_path (str): path to the destination file
        """
        file_obj = open(file_path, "w")
        line = ''
        delim = "\t"
        for i in range(len(list_objs)):
            for j in range(len(list_objs[i])):
                line = list_objs[i][j].chap_name + delim \
                + list_objs[i][j].sent_num + delim \
                + list_objs[i][j].token_num + delim \
                + list_objs[i][j].word + delim \
                + list_objs[i][j].lemma + delim \
                + list_objs[i][j].pos + delim \
                + list_objs[i][j].syntax
                
                if len(list_objs[i][j].negation_list) == 0:
                    line = line + delim + "***"
                else:
                    for elem in list_objs[i][j].negation_list:
                         line = line + delim + elem[0] + delim + elem[1]+ delim + elem[2]
            
                file_obj.write(line)
                file_obj.write("\n")
            
            #if i+1 < len(list_objs)  and int(list_objs[i+1].token_num) == 0:                
            file_obj.write("\n")
                
        file_obj.close()
    
    
    def split(self, org_file_path, file1_path, file2_path, parcent=None, count=None):
        """
        Split a file into two smaller files.
        :param org_file_path (str): path to the original file.
        :param file1_path (str): path to the first file.
        :param file2_path (str): path to the second file.
        :param parcent (str): percentage of sentences from the original will strore into the first file.
        :param count (str): number of sentences from the original will strore into the first file. 
                            If parcent variable is None then the count variable will not be checked.
        """
        
        data_objs = Dataprep().file_read(org_file_path)
        size_objs = len(data_objs)
        
        # Grouping tokens of a sentence together
        sent_objs = []
        obj_list  = []
        for i in range(size_objs):
            obj_list.append(data_objs[i])
            if (i == size_objs - 1) or (i + 1 < size_objs and int(data_objs[i + 1].token_num) == 0):
                sent_objs.append(obj_list)
                obj_list = []
        
        # Get indices lists for the two output splits
        size = len(sent_objs)
        if parcent:
            file1_num   = int(np.ceil(size * parcent))
            file1_indxs = list( np.random.choice(size, file1_num, replace=False) )
        else:
            assert size > count
            file1_indxs = list(np.random.choice(size, count, replace=False))
            
        file2_indxs = list(set(list(range(size))) - set(file1_indxs))
        
        print("Number of sentences: {}, File1 sentences: {}, File2 sentences: {}".format(size, len(file1_indxs), len(file2_indxs)))
        assert size == len(file1_indxs)+len(file2_indxs)
        
        file1_objs = [sent_objs[idx] for idx in file1_indxs]
        file2_objs = [sent_objs[idx] for idx in file2_indxs]
        
        self.save(file1_objs, file1_path)
        print("{} is crated.".format(file1_path))
        self.save(file2_objs, file2_path)
        print("{} is crated.".format(file2_path))
        
    def split_for_few_shot(self, org_file_path, file1_path, file2_path, train_count=None, dev_count=None):
        """
        Split a file into two smaller files.
        :param org_file_path (str): path to the original file.
        :param file1_path (str): path to the first file.
        :param file2_path (str): path to the second file.
        :param train_count (str): number of sentences from the original will store into the first file (as train data). 
        :param dev_count (str): number of sentences from the original will strore into the second file (as dev data).
        """
        
        data_objs = Dataprep().file_read(org_file_path)
        size_objs = len(data_objs)
        
        # Grouping tokens of a sentence together
        sent_objs = []
        obj_list  = []
        for i in range(size_objs):
            obj_list.append(data_objs[i])
            if (i == size_objs - 1) or (i + 1 < size_objs and int(data_objs[i + 1].token_num) == 0):
                sent_objs.append(obj_list)
                obj_list = []
                
                
        # Negated sentence objects only
        neg_sent_objs = []
        for sent in sent_objs:
            for obj in sent:
                if obj.negation_list:
                    neg_sent_objs.append(sent)
                    break
        
        
        # Get indices lists for the two output splits
        size = len(neg_sent_objs)
        assert size > train_count
        file1_indxs = list(np.random.choice(size, train_count, replace=False))
            
        remain_indxs = list(set(list(range(size))) - set(file1_indxs))
        assert len(remain_indxs) > dev_count
        file2_indxs = np.random.choice(remain_indxs, dev_count, replace=False)
        
        #validity check. No common indices in training and dev corpus
        assert 0 == len( set(file1_indxs).intersection(set(file2_indxs)) )
        
        print("Total sentences: {}, Number of Neg sentences: {}, File1 sentences: {}, File2 sentences: {}".format(len(sent_objs), size, len(file1_indxs), len(file2_indxs)))
        
        file1_objs = [neg_sent_objs[idx] for idx in file1_indxs]
        file2_objs = [neg_sent_objs[idx] for idx in file2_indxs]
        
        self.save(file1_objs, file1_path)
        print("{} is crated.".format(file1_path))
        self.save(file2_objs, file2_path)
        print("{} is crated.".format(file2_path))
        
        

class DataprepFile():
    def read_file(self, file_path, delim="\t"):
        """
        Read a file. File should contains sentences in the last column. 
        Each line contains exactly one sentence.
        Features are separated with tab if the file contains multiple features.
        :param file_path: path to the input file
        """
        sents = []
        with open(file_path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = re.sub("\n|\r", "", line.strip())
                sents.append(line)                                      
        return sents
        
    def preprocess(self, file_path, delim="\t"):
        """
        Read the sentences in the file to and preprocess so that cue-detector can read the data.
        :param file_path: path to the input file
        :param delim: delimiter in each line of the input file.
        """
        
        sents = self.read_file(file_path, delim)
        #print("sents: {}".format(sents))
        data = defaultdict(list)
        for sent in sents:         
            sent_tokens = [token.text for token in nlp(sent)]
            #print(sent_tokens)
            num_tokens = len(sent_tokens)
            data["sent"].append(sent)
            data["tokens"].append(sent_tokens)
            data["cues"].append(["O_C" for _ in range(num_tokens)])
            data["scopes"].append(["O_S" for _ in range(num_tokens)])
            data["events"].append(["O_E" for _ in range(num_tokens)])
            data["cue_positions"].append([])
            
        return data


class Connl_format():

    def get_cue(self, word):
        word_ = word.strip().lower()
        prefixs = ['dis', 'im', 'in', 'ir', 'un', 'non']
        suffixs = ['less', 'lessly', 'lessness']
        for prefix in prefixs:
            if word_.startswith(prefix) and len(prefix) < len(word):
                return prefix
        for suffix in suffixs:
            if word_.endswith(suffix) and len(suffix) < len(word):
                return suffix
        return word


    def prepare_connl_file(self, data_objs, pred_positions, file_path):
        """

        :param data_objs: List of lines of a connl format file
        :param pred_positions: positions of negation cues.
         For example, sentence 2 might have 2 cues, first cue resides in 8th and 9th positions, another resides in 14th positions, therefore pred_positions[2] = [[8,9], [14]]
        :param file_path: path to the connl output file
        :return:
        """

        # Grouping tokens of a sentence together
        size_objs = len(data_objs)
        sent_objs = []
        obj_list = []
        for i in range(size_objs):
            obj_list.append(data_objs[i])
            if (i == size_objs - 1) or (i + 1 < size_objs and int(data_objs[i + 1].token_num) == 0):
                sent_objs.append(obj_list)
                obj_list = []

        assert len(sent_objs) == len(pred_positions)

        # save into file
        with open(file_path, "w") as file_obj:
            delim = "\t"
            for i in range(len(sent_objs)): # i iterates over all sentences
                num_cues = len(pred_positions[i])
                cue_pos_dict = {}
                for idx, positions in enumerate(pred_positions[i]): # pred_positions[i] can be like this [[8,,9], [16]]
                    for pos in positions:
                        cue_pos_dict[pos] = idx
                #print ("i: {}, cue_pos_dict: {}".format(i, cue_pos_dict))
                for j in range(len(sent_objs[i])): # j iterates over all tokens in ith sentence
                    negation_list = [("_", "_", "_") for _ in range(num_cues)]
                    line = sent_objs[i][j].chap_name + delim + sent_objs[i][j].sent_num + delim + sent_objs[i][j].token_num + delim + \
                       sent_objs[i][j].word + delim + sent_objs[i][j].lemma + delim + sent_objs[i][j].pos + delim + sent_objs[
                           i][j].syntax
                    if num_cues == 0:
                        line = line + delim + "***"
                    else:
                        tok_num = int(sent_objs[i][j].token_num)
                        if tok_num in cue_pos_dict:
                            cue = self.get_cue(sent_objs[i][j].word)
                            negation_list[cue_pos_dict[tok_num]] = (cue, "_", "_")

                        for negation in negation_list:
                            line = line + delim + negation[0] + delim + negation[1] + delim + negation[2]

                    file_obj.write(line)
                    file_obj.write("\n")
                file_obj.write("\n")

