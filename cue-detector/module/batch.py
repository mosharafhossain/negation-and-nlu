# -*- coding: utf-8 -*-

import torch
import random
import numpy as np

def prepare_batch_data(params, batch_tokens, tokenizer, device):
    """
    Prepare a data batch. 
    :param params:
    :param batch_tokens:
    :param tokenizer:
    :return:
    """
    
    batch_piece_idxs  = []
    batch_attn_masks   = []
    batch_piece_lens = []
    cur_max_len = -1
    
    for tokens in batch_tokens:
        pieces = []
        piece_lens = []
        for token in tokens:
            token_pieces = [p for p in tokenizer.tokenize(token) if p]
            if len(token_pieces) == 0:
                #print("token: {}, token_pieces: {}".format(token, token_pieces))
                continue
            
            if len(pieces) + len(token_pieces) + 2 <= params["max_length"]: #+2 because tokenizer.encode method add 2 additional tokens.
                pieces.extend(token_pieces)
                piece_lens.append(len(token_pieces))
            else:
                #print("\ntokens:   {}\n".format(len(pieces) + len(token_pieces)))
                break
        
        # Pad word pieces with special tokens
        # This function adds two additional numbers than the list of pieces
        # ref: https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.encode_plus
        piece_idxs = tokenizer.encode(pieces,
                                      add_special_tokens=True,
                                      max_length=params["max_length"], # This truncation is actually not needed because len(pieces)+2 can not exceeds params["max_length"]. 
                                      truncation=True)
        
        num_pad = params["max_length"] - len(piece_idxs)
        attn_mask = [1] * len(piece_idxs) + [0] * num_pad
        piece_idxs = piece_idxs + [0] * num_pad
        if len(piece_lens) > cur_max_len:
            cur_max_len = len(piece_lens)
        
        if params["use_multi_gpus"]:
            piece_lens = piece_lens + [0] * (params["max_length"] - len(piece_lens))  #for multi-gpu
        
        batch_piece_idxs.append(piece_idxs)
        batch_attn_masks.append(attn_mask)
        batch_piece_lens.append(piece_lens) 
        
        #print("num_pad: {}, pieces: {}, attn_mask: {} piece_idxs bef: {} piece_idxs aft: {}".format(num_pad, len(pieces), len(attn_mask), len(tt), len(piece_idxs)))
        
    #cur_max_len = max([ len(piece_lens) for piece_lens in batch_piece_lens] )
    batch_piece_idxs = torch.LongTensor(batch_piece_idxs)
    batch_attn_masks = torch.FloatTensor(batch_attn_masks)
    if params["use_multi_gpus"]:
        batch_piece_lens = torch.LongTensor(batch_piece_lens) #for multi-gpu
      
    if params["use_gpu"]:
        batch_piece_idxs = batch_piece_idxs.to(device)
        batch_attn_masks = batch_attn_masks.to(device)
        if params["use_multi_gpus"]:
            batch_piece_lens = batch_piece_lens.to(device)  #for multi-gpu

        
    batch_data = [batch_piece_idxs, batch_attn_masks, batch_piece_lens]
    #print("\nbatch_max_len: {}".format(cur_max_len))
    return batch_data, cur_max_len

def prepare_batch_labels(params, vocabs, batch_labels, max_len, device, label_name):
    """
    Prepare a label batch. 
    :param params:
    :param batch_labels:
    :param vocabs:
    :param label_name:
    :return:
    """
    
    batch_label_idxs = []
    vocab_stoi = vocabs[label_name]
    
    for labels in batch_labels:
        idx = [vocab_stoi[l] for l in labels]
        if len(idx) > max_len:
            idx = idx[0:max_len]
        else:
            num_pad = max_len - len(idx)
            idx = idx + [vocab_stoi["PAD"]] * num_pad
        batch_label_idxs.append(idx)
        
    batch_label_idxs = torch.LongTensor(batch_label_idxs)
    if params["use_gpu"]:
        batch_label_idxs = batch_label_idxs.to(device)
    
    return batch_label_idxs

  
class Batchprep():
    def get_a_batch(self, params, vocabs, tokenizer, data, data_size, batch_num, device, shuffle=False):
        """
        Generate a batch of Data and labels.
        :param params: contains all parameter

        :return:
        """
       
        order = list(range(data_size)) # generate a list that decides the order in which we go over the data.
        
        if shuffle:
            random.shuffle(order)
        
        for i in range(batch_num):
            start = i*params["batch_size"]
            end   = (i+1)*params["batch_size"] if i < batch_num-1 else data_size
            curr_order = order[start:end]

            if "tokens" in params["features"]:
                #batch_tokens = list(np.array(data["tokens"])[curr_order])
                batch_tokens = [data["tokens"][idx] for idx in curr_order]
                batch_data, cur_max_len =  prepare_batch_data(params, batch_tokens, tokenizer, device)                
                
            if "cues" in params["features"]:
                #batch_cues = list(np.array(data["cues"])[curr_order])
                batch_cues = [data["cues"][idx] for idx in curr_order]
                batch_cues_idx = prepare_batch_labels(params, vocabs, batch_cues, cur_max_len, device, label_name="cues")

            yield batch_data, batch_cues_idx