# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from transformers import (BertModel, 
                          AlbertModel, 
                          #MT5Model,
                          RobertaModel,
                          AutoModelForMaskedLM)
import torch.nn.functional as F
from torchcrf import CRF
from module.loss import CalcLoss


def token_lens_to_offsets(token_lens):
    """
    This method is adapted from OneIE framework. Link: http://blender.cs.illinois.edu/software/oneie/
    Map token lengths to first word piece indices, used by the sentence
    encoder.
    :param token_lens (list): token lengths (word piece numbers)
    :return (list): first word piece indices (offsets)
    """
    max_token_num = max([len(x) for x in token_lens])
    offsets = []
    for seq_token_lens in token_lens:
        seq_offsets = [0]
        for l in seq_token_lens[:-1]:
            seq_offsets.append(seq_offsets[-1] + l)
        offsets.append(seq_offsets + [-1] * (max_token_num - len(seq_offsets)))
    return offsets

def token_lens_to_idxs(token_lens):
    """
    This method is adapted from OneIE framework. Link: http://blender.cs.illinois.edu/software/oneie/
    Map token lengths to a word piece index matrix (for torch.gather) and a
    mask tensor.
    For example (only show a sequence instead of a batch):

    token lengths: [1,1,1,3,1]
    =>
    indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
    masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]

    Next, we use torch.gather() to select vectors of word pieces for each token,
    and average them as follows (incomplete code):

    outputs = torch.gather(bert_outputs, 1, indices) * masks
    outputs = bert_outputs.view(batch_size, seq_len, -1, self.bert_dim)
    outputs = bert_outputs.sum(2)

    :param token_lens (list): token lengths.
    :return: a index matrix and a mask tensor.
    """
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:        
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            #print("\ntoken_len: {}".format(token_len))
            seq_idxs.extend([i + offset for i in range(token_len)]
                            + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len

class DetectNeg(nn.Module):
    def __init__(self, params, vocabs):
        super(DetectNeg, self).__init__()        
        self.multi_piece   = params["multi_piece"]
        self.use_multi_gpus  = params["use_multi_gpus"]
        self.use_crf = params["use_crf"]
        self.use_extra_features = params["use_extra_features"]
        self.extra_feature_layer = params["extra_feature_layer"]
        self.trasformer_system = params["pt_system"]
        
        if self.use_extra_features:        
            self.bert_dim = params["bert_dim"] * 2
        else:
            self.bert_dim = params["bert_dim"]
        
        
        self.num_cue_labels = len(vocabs["cues"])     
        
        if self.trasformer_system == "mBERT-base":
            self.bert = BertModel.from_pretrained(params["mBERT-base-path"], output_hidden_states=True)
        elif self.trasformer_system == "XLM-RoBERTa-base":  #https://huggingface.co/transformers/model_doc/xlmroberta.html#
            self.bert = AutoModelForMaskedLM.from_pretrained(params["XLM-RoBERTa-base-path"], output_hidden_states=True)
        #elif self.trasformer_system == "mT5-base":  #https://huggingface.co/transformers/model_doc/xlmroberta.html#
        #    self.bert = MT5Model.from_pretrained(params["mT5-base-path"], output_hidden_states=True)
        elif self.trasformer_system == "albert":
            self.bert = AlbertModel.from_pretrained(params["albert-path"], output_hidden_states=True)
        elif self.trasformer_system == "RoBERTa-base":
            self.bert = RobertaModel.from_pretrained(params["RoBERTa-base-path"], output_hidden_states=True)
        
        print("Transformer is loaded!")
        self.bert_dropout = nn.Dropout(params["bert_dropout"])
        
        self.fc_cue   = nn.Linear(self.bert_dim , self.num_cue_labels, bias=params["linear_bias"])
        if self.use_crf:
            self.crf = CRF(self.num_cue_labels, batch_first=True)
            
        # loss functions
        self.cue_criteria = CalcLoss().cross_entropy_loss  
        
    def encode(self, piece_idxs, attention_masks, token_lens):        
        """Encode input sequences with BERT
        This method is adapted from OneIE framework. Link: http://blender.cs.illinois.edu/software/oneie/
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths
        """
        
        all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        if self.trasformer_system == "mBERT-base" or self.trasformer_system == "albert" or self.trasformer_system == "RoBERTa-base":
            bert_outputs = all_bert_outputs[0]
            #print("1. bert_outputs dim: {}".format( bert_outputs.size() ) )
            if self.use_extra_features:
                extra_rep = all_bert_outputs[2][self.extra_feature_layer]
                #print("2. extra_rep dim: {}".format( extra_rep.size() ) )
                bert_outputs = torch.cat([bert_outputs, extra_rep], dim=2)
        elif self.trasformer_system == "XLM-RoBERTa-base" :
            bert_outputs = all_bert_outputs[1][-1]
            if self.use_extra_features:
                extra_rep = all_bert_outputs[1][self.extra_feature_layer]
                bert_outputs = torch.cat([bert_outputs, extra_rep], dim=2)
 
        if self.use_multi_gpus:
            token_lens = token_lens.tolist() #for multi-gpu
            token_lens = [[e for e in seq_token_lens if e!=0] for seq_token_lens in token_lens]         
        batch_size, _ = piece_idxs.size()
        
        if self.multi_piece == 'first':
            # select the first piece for multi-piece words
            offsets = token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets)
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            bert_outputs = torch.gather(bert_outputs, 1, offsets)
        elif self.multi_piece == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
            idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            masks = bert_outputs.new(masks).unsqueeze(-1)
            
            #print("0.1. bert_outputs dim: {}".format( bert_outputs.size() ) )
            #print("0.2. idxs dim: {}".format( idxs.size() ) )
            #print("0.3. masks dim: {}".format( masks.size() ) )
            bert_outputs = torch.gather(bert_outputs, 1, idxs) * masks
            bert_outputs = bert_outputs.view(batch_size, token_num, token_len, self.bert_dim)
            bert_outputs = bert_outputs.sum(2)
        else:
            raise ValueError('Unknown multi-piece token handling strategy: {}'
                             .format(self.multi_piece))
        bert_outputs = self.bert_dropout(bert_outputs)
        return bert_outputs
        
    def forward(self, batch_data, batch_labels):
        """
        """
        piece_idxs      = batch_data[0]
        attention_masks = batch_data[1]
        piece_lens      = batch_data[2]
        #print("\ncurrent gpu: {}, piece_idxs: {}, attention_masks: {}, piece_lens: {}".format(torch.cuda.current_device(), piece_idxs.size(), attention_masks.size(), piece_lens.size() ))
        
        X = self.encode(piece_idxs, attention_masks, piece_lens)      
        #print("1. X dim: {}".format( X.size() ) )
        batch_size, seq_len, _ = X.size()
        
        
        X = X.contiguous()           # X dim: batch_size x seq_len x hidden_size
        X = X.view(-1, X.shape[2])   # X dim: batch_size*seq_len x hidden_size
        #print("2. X dim: {}".format( X.size() ) )
        
        X = self.fc_cue(X) # X dim: batch_size*seq_len x num_labels, X = X.W, (batch_size*seq_len x lstm_units).(lstm_units x num_labels)
        #print("3. X dim: {}".format( X.size() ) )
        
                
        if self.use_crf:            
            X = X.contiguous()
            X = X.view(batch_size, seq_len, self.num_cue_labels) #Dim: batch_size x seq_len x num_labels
            #print("4. X dim: {}".format( X.size() ) )
            #print("5. batch_labels dim: {}".format( batch_labels[0:batch_size, 0:seq_len].size() ) )
            loss = -1.0 * self.crf(X, batch_labels[0:batch_size, 0:seq_len]) # Taking [0:batch_size, 0:seq_len], because batch_size and seq_len might be different for some situations.
            loss = loss/batch_size #average loss  
            
        else:
            X = F.log_softmax(X, dim=1)   # dim: batch_size*seq_len x num_labels
            loss = self.cue_criteria(X, batch_labels[0:batch_size, 0:seq_len])
        #print("loss: {}\n".format(loss))   

        return loss
    
    
    def predict(self, batch_data):
        """
        """
        self.eval()
        piece_idxs      = batch_data[0]
        attention_masks = batch_data[1]
        piece_lens      = batch_data[2]
        #print("\npiece_idxs: {}, attention_masks: {}, piece_lens: {}".format(piece_idxs.size(), attention_masks.size(), len(piece_lens) ))
        
        X = self.encode(piece_idxs, attention_masks, piece_lens)      
        #print("1. X dim: {}".format( X.size() ) )
        batch_size, seq_len, _ = X.size()
        
        X = X.contiguous()           # X dim: batch_size x seq_len x hidden_size
        X = X.view(-1, X.shape[2])   # X dim: batch_size*seq_len x hidden_size
        #print("2. X dim: {}".format( X.size() ) )
        
        X = self.fc_cue(X) # X dim: batch_size*seq_len x num_labels, X = X.W, (batch_size*seq_len x lstm_units).(lstm_units x num_labels)
        #print("3. X dim: {}".format( X.size() ) )
        
                
        if self.use_crf:
            X = X.contiguous()
            X = X.view(batch_size, seq_len, self.num_cue_labels) #Dim: batch_size x seq_len x num_labels
            #print("4. X dim: {}".format( X.size() ) )
            X = self.crf.decode(X) # dim (list): batch_size x seq_len
            
        else:
            X = F.log_softmax(X, dim=1)   # dim: batch_size*seq_len x num_labels            
            X = X.contiguous()           # X dim: batch_size * seq_len x num_labels
            X = X.view(batch_size, seq_len, self.num_cue_labels)   # X dim: batch_size x seq_len x num_labels
            #print("4. X dim: {}".format( X.size() ) )
            X = torch.argmax(X, dim=2)  
            X = X.tolist() #dim (list): batch_size x seq_len
        
        self.train()
        return X
    



