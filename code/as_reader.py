# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:07:44 2017

@author: lcr
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import as_layers
import numpy as np
import mt_lstm


class ASReader(nn.Module):
    """QA module"""
    def __init__(self,
                 opts,
                 pos_size,
                 ner_size,
                 char_vocab_size,
                 kernel_sizes,
                 vocab_size,
                 que_vocab_size,
                 init_embeddings,
                 fixed_embeddings,
                 qix_to_aix,
                 sos_idx,
                 padding_idx=0):
        super(ASReader,self).__init__()
        self.opts = opts
        self.hidden_dim = opts.hidden_dim
        self.embedding_dim = opts.embedding_dim
        self.feature_dim = opts.extra_feature_num
        self.pos_embedding_dim = opts.pos_dim
        self.ner_embedding_dim = opts.ner_dim
        self.question_type_dim = opts.question_type_dim
        self.char_embedding_dim = opts.char_embedding_dim
        self.char_embedding_hidden_dim = opts.char_embedding_hidden_dim
        self.cove_embedding_dim = opts.cove_embedding_dim
        self.que_type_dim = opts.question_type_dim
        self.que_type_num = opts.question_type_num
        self.embedding = nn.Embedding(vocab_size,
                                      self.embedding_dim,
                                      padding_idx=padding_idx)
        self.pos_embedding = nn.Embedding(pos_size,
                                          self.pos_embedding_dim,
                                          padding_idx=padding_idx)
        self.ner_embedding = nn.Embedding(ner_size,
                                          self.ner_embedding_dim,
                                          padding_idx=padding_idx)
        self.que_type_embedding = nn.Embedding(self.que_type_num,
                                               self.que_type_dim)
                                          
        self.embedding.weight = nn.Parameter(init_embeddings)
        self.embedding.weight.requires_grad = True
        self.fixed_embeddings =fixed_embeddings
        self.fixed_num = opts.fixed_embedding_num
        self.hop_nums = opts.hop_nums
        self.vocab_size = vocab_size
        self.que_vocab_size = que_vocab_size
        self.rnn_layers_num = opts.rnn_layer_num
        self.char_embedding_use = opts.char_embedding_use
        self.cove_embedding_use = opts.cove_embedding_use
        self.word_dropout_use = opts.word_dropout_use
        self.word_dropout_rate = opts.word_dropout_rate
        self.dropout_rate = opts.dropout_rate
        self.question_generate_use = opts.question_generate_use
        self.adversarial_training = opts.adversarial_training
        self.question_classification_use = opts.question_classification_use
        self.qix_to_aix = qix_to_aix
        #doc_to_question representation
        self.doc_to_question = as_layers.SeqAttAlignment(self.embedding_dim)
#        self.doc_to_question_cove = as_layers.SeqAttAlignment(2*self.cove_embedding_dim)
        #char_embedding encoding representation
        self.char_embedding_cnn = as_layers.CharEmbedding(vocab_size=char_vocab_size,
                                                          input_size=self.char_embedding_dim,
                                                          out_channels=self.char_embedding_hidden_dim,
                                                          kernel_size=kernel_sizes,
                                                          dropout_rate=self.dropout_rate) 
        self.cove = mt_lstm.MTLSTM(n_vocab=vocab_size, vectors=init_embeddings)
        for para in self.cove.parameters():
            para.requires_grad = False
        if opts.char_embedding_use:
            self.doc_input_dim = 2*self.embedding_dim+self.feature_dim+self.pos_embedding_dim+self.ner_embedding_dim+2*self.char_embedding_hidden_dim 
            question_input_dim = self.embedding_dim+2*self.char_embedding_hidden_dim
        else:
            self.doc_input_dim = 2*self.embedding_dim+self.feature_dim+self.pos_embedding_dim+self.ner_embedding_dim
            question_input_dim = self.embedding_dim
        if opts.cove_embedding_use:
            self.doc_input_dim += 2*self.cove_embedding_dim
            question_input_dim += 2*self.cove_embedding_dim
            
        #encoding doc in bi-lstm
        self.doc_rnn = as_layers.StackedBiRNN(input_size=self.doc_input_dim,
                                              hidden_dim=self.hidden_dim,
                                              rnn_layers_num=self.rnn_layers_num,
                                              dropout_rate=self.dropout_rate,
                                              padding=False)
        
        #encoding question in bi-lstm
        self.question_rnn = as_layers.StackedBiRNN(input_size=question_input_dim,
                                                   hidden_dim=self.hidden_dim,
                                                   rnn_layers_num=self.rnn_layers_num,
                                                   dropout_rate=self.dropout_rate,
                                                   padding=False)
        self.doc_hidden_dim = 2*self.hidden_dim*self.rnn_layers_num
        self.question_hidden_dim = 2*self.hidden_dim*self.rnn_layers_num

        #question-aligned weight
        self.aligned_att = as_layers.LinearSeqAtt(self.question_hidden_dim)
        
        #self attention 
        self.self_att = as_layers.SelfAttention(self.doc_hidden_dim)
        
        #self attention2 
        self.self_att2 = as_layers.SelfAttention(self.doc_hidden_dim)
        
        #doc_to_question representation2
        self.doc_to_question2 = as_layers.SeqAttAlignment(self.doc_hidden_dim)
        
        #doc_to_question representation3
        self.doc_to_question3 = as_layers.SeqAttAlignment(self.doc_hidden_dim)
        
        #encoding doc in bi-lstm2
        self.doc_rnn2 = as_layers.StackedBiRNN(input_size=3*self.doc_hidden_dim,
                                               hidden_dim=self.hidden_dim,
                                               rnn_layers_num=self.rnn_layers_num,
                                               dropout_rate=self.dropout_rate,
                                               padding=False)

        #encoding doc in bi-lstm3
        self.doc_rnn3 = as_layers.StackedBiRNN(input_size=3*self.doc_hidden_dim,
                                               hidden_dim=self.hidden_dim,
                                               rnn_layers_num=self.rnn_layers_num,
                                               dropout_rate=self.dropout_rate,
                                               padding=False)
                                               
        #seq2doc attention
        self.seq2doc = as_layers.Seq2DocAtt(x_dim=self.doc_hidden_dim,
                                            y_dim=self.question_hidden_dim)
                                            
        #MLP
        self.MLP = as_layers.MLPModule(input_size=self.question_hidden_dim+self.doc_hidden_dim,
                                       output_size=self.question_hidden_dim)       
        
        #count the score for start index
        self.start_predict = as_layers.BilinearSeqAtt(self.doc_hidden_dim,
                                                      self.question_hidden_dim)
        #count the score for end index
        self.end_predict = as_layers.BilinearSeqAtt(self.doc_hidden_dim,
                                                    self.question_hidden_dim)
        self.answer_output = as_layers.AnswerModule(self.doc_hidden_dim,
                                                    self.question_hidden_dim,
                                                    self.hop_nums)
        
                                                
    def forward(self,x,xc,x_f,x_mask,y,yc,yt,y_mask,answer_idx):
        '''        
        input:
            x: (batch_size,doc_size)
            xc: (batch_size,doc_size,doc_word_size)
            x_f: (batch_size,doc_size,designed_feature_num)
            x_mask: (batch_size,doc_size)
            y: (batch_size,question_size)
            yc: (batch_size,question_size,question_word_size)
            yt: (batch_size,question_type_num) 未使用
            y_mask: (batch_size)
            answere_idx: list[0] is start index, list[1] is end index
        return:
            start_scores: (batch_size,doc_size)
            end_scores: (batch_size,doc_size)
        '''
        if self.word_dropout_use and self.training:
            x = self.word_dropout(x,self.word_dropout_rate)
            y = self.word_dropout(y,self.word_dropout_rate)
        x_emb = self.embedding(x)#[batch_size*doc_size*embedding_dim]
        y_emb = self.embedding(y)#[batch_size*question_size*embedding_dim]
        pos_emb = self.pos_embedding(x_f[1])
        ner_emb = self.ner_embedding(x_f[2])
        if self.dropout_rate > 0:
            x_emb = F.dropout(x_emb,
                              p=self.dropout_rate,
                              training=self.training)
            y_emb = F.dropout(y_emb,
                              p=self.dropout_rate,
                              training=self.training)
            pos_emb = F.dropout(pos_emb,
                                p=self.dropout_rate,
                                training=self.training)
            ner_emb = F.dropout(ner_emb,
                                p=self.dropout_rate,
                                training=self.training)
        #weighted_question representation for doc
        weighted_y1 = self.doc_to_question(x_emb,y_emb,y_mask)#[batch_size*doc_size*embedding_dim]

        question_rnn_input = y_emb
        if self.char_embedding_use:
            yc_emb = self.char_embedding_cnn(yc)
            question_rnn_input = torch.cat([question_rnn_input,yc_emb],2)
        
        if self.cove_embedding_use:
            y_cove = self.cove(y,y_mask)
            question_rnn_input = torch.cat([question_rnn_input,y_cove],2)
        
        #lstm encoding for question_input
        questions_encoding,que_hidden,que_encoding_last = self.question_rnn(question_rnn_input,y_mask)

#-------------------------------------------------------------------------------
        #layer1
        doc_rnn_input1 = torch.cat([x_emb,weighted_y1,x_f[0],pos_emb,ner_emb],2)
        
        if self.char_embedding_use:
            xc_emb = self.char_embedding_cnn(xc)
            doc_rnn_input1 = torch.cat([doc_rnn_input1,xc_emb],2)
        
        if self.cove_embedding_use:
            x_cove = self.cove(x,x_mask)
            doc_rnn_input1 = torch.cat([doc_rnn_input1,x_cove],2)

        #lstm encoding for doc_input
        doc_encoding1,doc_hidden1,doc_encoding_last1 = self.doc_rnn(doc_rnn_input1,x_mask)
#--------------------------------------------------------------------------------
        #layer2          
        #self-attention representation for doc
        self_att_doc1 = self.self_att(doc_encoding1,x_mask)

        #weighted question representation for doc2
        weighted_y2 = self.doc_to_question2(doc_encoding1,questions_encoding,y_mask) 
        
        doc_rnn_input2 = torch.cat([doc_encoding1,weighted_y2,self_att_doc1],2)
#        doc_rnn_input2 = torch.cat([doc_encoding1,weighted_y2],2)
#        doc_rnn_input2 = torch.cat([doc_encoding1,self_att_doc1],2)
      
        doc_encoding2,doc_hidden2,doc_encoding_last2 = self.doc_rnn2(doc_rnn_input2,x_mask)
#--------------------------------------------------------------------------------
#        #layer3
        self_att_doc2 = self.self_att2(doc_encoding2,x_mask)

        weighted_y3 = self.doc_to_question3(doc_encoding2,questions_encoding,y_mask)
        
        doc_rnn_input3 = torch.cat([doc_encoding2,weighted_y3,self_att_doc2],2)
#        doc_rnn_input3 = torch.cat([doc_encoding2,weighted_y3],2)
#        doc_rnn_input3 = torch.cat([doc_encoding2,self_att_doc2],2)
        doc_encoding3,doc_hidden3,doc_encoding_last3 = self.doc_rnn3(doc_rnn_input3,x_mask)
#--------------------------------------------------------------------------------
        #combine
#        doc_encoding = torch.cat([doc_encoding1,doc_encoding2,doc_encoding3],2)
#        doc_encoding = torch.cat([doc_encoding1,doc_encoding2],2)
        doc_encoding = doc_encoding3
#        doc_encoding,_,_ = self.doc_rnn_last(doc_encoding,x_mask)
#--------------------------------------------------------------------------------                        
        #question merging
        question_att = self.aligned_att(questions_encoding,y_mask)
        question_encoding = as_layers.weighted_avg(questions_encoding,question_att)
        
        #queSeq2doc encoding
        weighted_question_encoding = self.seq2doc(doc_encoding,question_encoding,x_mask)
        
        question_encoding = torch.cat([question_encoding,weighted_question_encoding],1)
        question_encoding = self.MLP(question_encoding)

#---------------------------------------------------------------------------------        
        #predicting the start and end
        start_scores,start_soft = self.start_predict(doc_encoding,question_encoding,x_mask)
        end_scores,end_soft = self.end_predict(doc_encoding,question_encoding,x_mask)
#        scores,soft = self.answer_output(doc_encoding,question_encoding,x_mask)
#        start_scores =scores[0]
#        end_scores = scores[1]
#        start_soft = soft[0]
#        end_soft = soft[1]
#--------------------------------------------------------------------------------- 
        
        
#        if self.opts.question_generate_use:
##            answer_encoding = self.get_ans_representation(start_scores,end_scores,doc_encoding)
#            doc_proj = self.MLP_QQ(doc_encoding)
#            answer_encoding = self.get_ans_repreentation_soft_add(start_soft,end_soft,doc_proj)
##            answer_encoding_gold = self.get_ans_representation_teacherforcing_add(answer_idx[0],answer_idx[1],doc_encoding3)
#            seq_scores = self.seq_decoder_rnn(doc_encoding,answer_encoding,y.size(1))
##            seq_scores_gold = self.seq_decoder_rnn(doc_encoding,answer_encoding_gold,y.size(1))
        
        if self.opts.adversarial_training and self.training:               
            return start_scores,end_scores,doc_encoding,[start_soft,end_soft]
        else:
            return start_scores,end_scores
            
        
    def word_dropout(self,x,prob):
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                p = np.random.random()
                if (p < prob):
                    x.data[i][j] = 0
        return x
        
    def reset_embeddings(self):
        self.embedding.weight.data[self.fixed_num+3:] = self.fixed_embeddings
        
class Discriminator(nn.Module):
    def __init__(self,opts,doc_hidden_dim,vocab_size,que_vocab_size,qix_to_aix,sos_idx,init_embeddings,padding_idx=0):
        super(Discriminator,self).__init__()
        self.opts = opts
        self.embedding_dim = opts.embedding_dim
        self.hidden_dim = opts.hidden_dim
        self.embedding = nn.Embedding(vocab_size,
                                      self.embedding_dim,
                                      padding_idx=padding_idx)
        self.embedding.weight.data = init_embeddings.weight.data
#        self.embedding.weight.requires_grad = False
        self.vocab_size = vocab_size
        self.que_vocab_size = que_vocab_size
        self.dropout_rate = opts.dropout_rate
        self.qix_to_aix = qix_to_aix
        #MLP for QG initial input
        self.MLP_QQ = as_layers.MLPModule(input_size=doc_hidden_dim,
                                          output_size=2*self.hidden_dim)
        #question sequence module
        self.seq_decoder_rnn = as_layers.AttDecoder(vocab_size=self.que_vocab_size,
#                                                    input_size=self.embedding_dim + 12*self.hidden_dim,
#                                                    input_size=self.embedding_dim + 6*self.hidden_dim,
                                                    input_size=self.embedding_dim,
                                                    hidden_size=2*self.hidden_dim,
                                                    embedding=self.embedding,
                                                    dropout_rate=self.dropout_rate,
                                                    qix_to_aix=self.qix_to_aix,
                                                    sos_idx=sos_idx)
                                                    
    
    def reset_embedding(self,input_embeddings):
        self.embedding.weight.data = input_embeddings.weight.data
                                                
    def forward(self,doc_encoding,score_soft,answer_idx,que_length):
        start_soft = score_soft[0]
        end_soft = score_soft[1]
        doc_proj = self.MLP_QQ(doc_encoding)
        answer_encoding = self.get_ans_representation_hard_span(start_soft,end_soft,doc_proj)
        answer_encoding_gold = self.get_ans_representation_teacherforcing_span(answer_idx[0],answer_idx[1],doc_proj)
        seq_scores = self.seq_decoder_rnn(doc_encoding,answer_encoding,que_length)
        seq_scores_gold = self.seq_decoder_rnn(doc_encoding,answer_encoding_gold,que_length)
        return seq_scores,seq_scores_gold
        
        
    def get_ans_representation_teacherforcing_add(self,s_idx,e_idx,doc_encoding):
        s_hidden = doc_encoding[range(doc_encoding.size(0)),s_idx, : ] #[batch_size,2*hidden_dim]
        e_hidden = doc_encoding[range(doc_encoding.size(0)),e_idx, : ] #[batch_size,2*hidden_dim]
        res = s_hidden + e_hidden
        return res

    def get_ans_representation_teacherforcing_cat(self,s_idx,e_idx,doc_encoding):
        s_hidden = doc_encoding[range(doc_encoding.size(0)),s_idx, : ] #[batch_size,2*hidden_dim]
        e_hidden = doc_encoding[range(doc_encoding.size(0)),e_idx, : ] #[batch_size,2*hidden_dim]
        res = torch.cat([s_hidden,e_hidden],1)
        return res
        
    def get_ans_representation_teacherforcing_span(self,s_idx,e_idx,doc_encoding):
        answer_encodings = []
        for i in range(doc_encoding.size(0)):
            answer_encoding = torch.sum(doc_encoding[i][s_idx[i]:e_idx[i]+1],0).unsqueeze(0)
            answer_encodings.append(answer_encoding)
        answer_encodings = torch.cat(answer_encodings,0)
        return answer_encodings
            
    def get_ans_representation(self,start_scores,end_scores,doc_encoding):
        s_idx = start_scores.max(1)[1] #[batch_size]
        e_idx = end_scores.max(1)[1] #[batch_size]
        s_hidden = doc_encoding[range(doc_encoding.size(0)),s_idx.data, : ] #[batch_size,2*hidden_dim]
        e_hidden = doc_encoding[range(doc_encoding.size(0)),e_idx.data, : ] #[batch_size,2*hidden_dim]
        s_hidden_forward = s_hidden[ : , :self.hidden_dim]
        s_hidden_backward = s_hidden[ : ,self.hidden_dim: ]
        e_hidden_forward = e_hidden[ : , : self.hidden_dim]
        e_hidden_backward = e_hidden[ : ,self.hidden_dim: ]
        res_forward = e_hidden_forward - s_hidden_forward
        res_backward = s_hidden_backward - e_hidden_backward
        res = torch.cat([res_forward,res_backward],1)
        for i in range(len(res)):
            if s_idx[i].data[0] < e_idx[i].data[0]: 
                res[i].data = torch.zeros(res.size(1))
#        res = []
#        for i in range(self.rnn_layers_num):
#            s_hidden_i = s_hidden[ : ,i*2*self.hidden_dim:(i+1)*2*self.hidden_dim]
#            e_hidden_i = e_hidden[ : ,i*2*self.hidden_dim:(i+1)*2*self.hidden_dim]
#            s_hidden_forward_i = s_hidden_i[ : , :self.hidden_dim]
#            s_hidden_backward_i = s_hidden_i[ : , self.hidden_dim: ]
#            e_hidden_forward_i = e_hidden_i[ : , :self.hidden_dim]
#            e_hidden_backward_i = e_hidden_i[ : ,self.hidden_dim: ]
#            forward_i = e_hidden_forward_i - s_hidden_forward_i #[batch_size,hidden_dim]
#            backward_i = s_hidden_backward_i - e_hidden_backward_i #[batch_size,hidden_dim]
#            res_i = torch.cat([forward_i,backward_i],1) #[batch_size,2*hidden_dim]
#            res.append(res_i)
#        ress = torch.cat(res,1) #[batch_size,6*hidden_dim]
#        for i in range(len(ress)):
#            if s_idx[i].data[0] < e_idx[i].data[0]:
#                ress[i].data = torch.zeros(ress.size(1))
        return res
        
    def get_ans_representation_hard_add(self,start_scores,end_scores,doc_encoding):
        s_idx = start_scores.max(1)[1] #[batch_size]
        e_idx = end_scores.max(1)[1] #[batch_size]
        s_hidden = doc_encoding[range(doc_encoding.size(0)),s_idx.data, : ] #[batch_size,2*hidden_dim]
        e_hidden = doc_encoding[range(doc_encoding.size(0)),e_idx.data, : ] #[batch_size,2*hidden_dim]
        res = s_hidden + e_hidden
        for i in range(len(res)):
            if s_idx[i].data[0] < e_idx[i].data[0]: 
                res[i].data = torch.zeros(res.size(1))
        return res
        
    def get_ans_representation_hard_cat(self,start_scores,end_scores,doc_encoding):
        s_idx = start_scores.max(1)[1] #[batch_size]
        e_idx = end_scores.max(1)[1] #[batch_size]
        s_hidden = doc_encoding[range(doc_encoding.size(0)),s_idx.data, : ] #[batch_size,2*hidden_dim]
        e_hidden = doc_encoding[range(doc_encoding.size(0)),e_idx.data, : ] #[batch_size,2*hidden_dim]
        res = torch.cat([s_hidden,e_hidden],1)
        for i in range(len(res)):
            if s_idx[i].data[0] < e_idx[i].data[0]: 
                res[i].data = torch.zeros(res.size(1))
        return res
        
    def get_ans_representation_hard_span(self,start_scores,end_scores,doc_encoding):
        s_idx = start_scores.max(1)[1]
        e_idx = end_scores.max(1)[1]
        answer_encodings = []
        for i in range(doc_encoding.size(0)):
            if s_idx[i].data[0] > e_idx[i].data[0]:
                answer_encoding = Variable(torch.zeros(1,doc_encoding.size(2)))
                if torch.cuda.is_available():
                    answer_encoding = answer_encoding.cuda()
            else:
                answer_encoding = torch.sum(doc_encoding[i][s_idx[i].data[0]:e_idx[i].data[0]+1],0).unsqueeze(0)
            answer_encodings.append(answer_encoding)
        answer_encodings = torch.cat(answer_encodings,0)
        return answer_encodings
        
    def get_ans_representation_soft(self,start_scores,end_scores,doc_encoding):
        '''
        start_scores:(batch_size,doc_size)
        end_scores:(batch_size,doc_size)
        doc_encoding:(batch_size,doc_size,hidden_size)
        '''
        start_scores = start_scores.unsqueeze(1)
        end_scores = end_scores.unsqueeze(1)
        s_hidden = start_scores.bmm(doc_encoding).squeeze(1) #[batch_size,2*hidden_dim]
        e_hidden = end_scores.bmm(doc_encoding).squeeze(1) #[batch_size,2*hidden_dim]
        s_hidden_forward = s_hidden[ : , :self.hidden_dim]
        s_hidden_backward = s_hidden[ : ,self.hidden_dim: ]
        e_hidden_forward = e_hidden[ : , : self.hidden_dim]
        e_hidden_backward = e_hidden[ : ,self.hidden_dim: ]
        res_forward = e_hidden_forward - s_hidden_forward
        res_backward = s_hidden_backward - e_hidden_backward
        res = torch.cat([res_forward,res_backward],1)
#        res = []
#        for i in range(self.rnn_layers_num):
#            s_hidden_i = s_hidden[ : ,i*2*self.hidden_dim:(i+1)*2*self.hidden_dim]
#            e_hidden_i = e_hidden[ : ,i*2*self.hidden_dim:(i+1)*2*self.hidden_dim]
#            s_hidden_forward_i = s_hidden_i[ : , :self.hidden_dim]
#            s_hidden_backward_i = s_hidden_i[ : , self.hidden_dim: ]
#            e_hidden_forward_i = e_hidden_i[ : , :self.hidden_dim]
#            e_hidden_backward_i = e_hidden_i[ : ,self.hidden_dim: ]
#            forward_i = e_hidden_forward_i - s_hidden_forward_i #[batch_size,hidden_dim]
#            backward_i = s_hidden_backward_i - e_hidden_backward_i #[batch_size,hidden_dim]
#            res_i = torch.cat([forward_i,backward_i],1) #[batch_size,2*hidden_dim]
#            res.append(res_i)
#        ress = torch.cat(res,1) #[batch_size,6*hidden_dim]
        return res
        
    def get_ans_repreentation_soft_add(self,start_scores,end_scores,doc_encoding):
        start_scores = start_scores.unsqueeze(1)
        end_scores = end_scores.unsqueeze(1)
        s_hidden = start_scores.bmm(doc_encoding).squeeze(1) #[batch_size,2*hidden_dim]
        e_hidden = end_scores.bmm(doc_encoding).squeeze(1) #[batch_size,2*hidden_dim]
        res = s_hidden+e_hidden
        return res
    
    def get_ans_repreentation_soft_cat(self,start_scores,end_scores,doc_encoding):
        start_scores = start_scores.unsqueeze(1)
        end_scores = end_scores.unsqueeze(1)
        s_hidden = start_scores.bmm(doc_encoding).squeeze(1) #[batch_size,2*hidden_dim]
        e_hidden = end_scores.bmm(doc_encoding).squeeze(1) #[batch_size,2*hidden_dim]
        res = torch.cat([s_hidden,e_hidden],1)
        return res