# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 18:58:22 2017

@author: lcr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

     
class StackedBiRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_dim,
                 rnn_layers_num,
                 dropout_rate,
                 padding=False):
        '''
        input_size: the dim of rnn vector input
        hidden_dim: the hidden dim of single-diretional rnn output 
        rnn_layers_num: the layer num of rnn
        padding: whether padding? training(unpadded), test(padded)
        '''
        super(StackedBiRNN,self).__init__()
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.rnn_layers_num = rnn_layers_num
        self.rnns = nn.ModuleList()
        for i in range(rnn_layers_num):
            input_size = input_size if i == 0 else 2*hidden_dim
            self.rnns.append(nn.LSTM(input_size,
                                     hidden_dim,
                                     num_layers=1,
                                     batch_first=True,
                                     bidirectional=True))
                                     
    def forward(self,x,x_mask):
        if x_mask.data.sum() == 0:
            return self.lstm_encoding_unpadded(x,x_mask)
        if self.padding or not self.training:
            return self.lstm_encoding(x,x_mask)
        return self.lstm_encoding_unpadded(x,x_mask)
        
    def lstm_encoding_unpadded(self,x,x_mask):
        '''
        input:
            x: (batch_size,sen_size,input_size)
            x_mask: (batch_size,sen_size)
        return:
            output: (batch_size,sen_size,lay_num*direction_num*hidden_dim)
            hidden: tuple(h,c)
            output_last: (batch_size,sen_size,direction_hidden_dim)
        '''
        outputs = [x]
        hs = []
        cs = []
        for i in range(self.rnn_layers_num):
            rnn_input = outputs[-1]
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            rnn_outputs,(h,c) =self.rnns[i](rnn_input)
            outputs.append(rnn_outputs) #[batch_size*sen_size*hidden_dim]
            rnn_h = torch.cat([h[-1],h[-2]],1).unsqueeze(0) #[1*batch_size*(direction_num*hidden_dim)]
            rnn_c = torch.cat([c[-1],c[-2]],1).unsqueeze(0) #[1*batch_size*(direction_num*hidden_dim)]
            hs.append(rnn_h)
            cs.append(rnn_c)
        output = torch.cat(outputs[1:],2)#[batch_size*sen_size*(lay_num*direction_num*hidden_dim)]
        output_last = outputs[-1]
        h = torch.cat(hs,0) #[layer_num*batch_size*(direction_num*hidden_dim)]
        c = torch.cat(cs,0) #[layer_num*batch_size*(direction_num*hidden_dim)]

        if self.dropout_rate > 0:
            output = F.dropout(output,p=self.dropout_rate,training=self.training)
            output_last = F.dropout(output_last,p=self.dropout_rate,training=self.training)
            h = F.dropout(h,p=self.dropout_rate,training=self.training)
            c = F.dropout(c,p=self.dropout_rate,training=self.training)
        hidden = (h,c)        
        return output,hidden,output_last
        
    def lstm_encoding(self,x,x_mask):
        '''
        input:
            x: (batch_size,sen_size,input_size)
            x_mask: (batch_size,sen_size)
        return:
            output: (batch_size,sen_size,lay_num*direction_num*hidden_dim)
            hidden: tuple(h,c)
            output_last: (batch_size,sen_size,direction_hidden_dim)
        '''
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _,idx_sort = torch.sort(lengths,dim=0,descending=True)
        _,idx_unsort = torch.sort(idx_sort,dim=0)
        
        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)
        
        x = x.index_select(0,idx_sort)
        
        rnn_input = nn.utils.rnn.pack_padded_sequence(x,lengths,batch_first=True)
        outputs = [rnn_input]
        hs = []
        cs = []
        for i in range(self.rnn_layers_num):
            rnn_input = outputs[-1]
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            rnn_output,(h,c) = self.rnns[i](rnn_input)
            rnn_h = torch.cat([h[-1],h[-2]],1).unsqueeze(0) #[1*batch_size*(direction_num*hidden_dim)]
            rnn_c = torch.cat([c[-1],c[-2]],1).unsqueeze(0) #[1*batch_size*(direction_num*hidden_dim)]
            hs.append(rnn_h)
            cs.append(rnn_c)
            outputs.append(rnn_output)
        
        for i,o in enumerate(outputs[1:],1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o,batch_first=True)[0]
        
        output = torch.cat(outputs[1:],2)
        output_last = outputs[-1]
        output = output.index_select(0,idx_unsort)
        output_last = output_last.index_select(0,idx_unsort)
        h = torch.cat(hs,0) #[layer_num*batch_size*(direction_num*hidden_dim)]
        c = torch.cat(cs,0) #[layer_num*batch_size*(direction_num*hidden_dim)]
        h = h.index_select(1,idx_unsort)
        c = c.index_select(1,idx_unsort)
        if self.dropout_rate > 0:
            output = F.dropout(output,p=self.dropout_rate,training=self.training)
            output_last = F.dropout(output_last,p=self.dropout_rate,training=self.training)
            h = F.dropout(h,p=self.dropout_rate,training=self.training)
            c = F.dropout(c,p=self.dropout_rate,training=self.training)
        hidden = (h,c)

        return output,hidden,output_last
        
class SeqAttAlignment(nn.Module):
    """x-to-y attention representation module"""
    def __init__(self,input_size,identity=False):
        super(SeqAttAlignment,self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size,input_size)
        else:
            self.linear = None
    
    def forward(self,x,y,y_mask):
        '''
        input:
            x:(batch_size,xsen_size,x_feature_dim)
            y:(batch_size,ysen_size,y_feature_dim)
            y:(batch_size,ysen_size)
        return:
            weighted_question:(batch_size,xsen_size,y_feature_dim)
        '''
        if self.linear:
            x_proj = self.linear(x.view(-1,x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1,y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y
        
        scores = x_proj.bmm(y_proj.transpose(1,2))#[batch_size*doc_size*question_size]
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data,-float('inf'))
        
        y_att=F.softmax(scores.view(-1,y.size(1)))
        y_att = y_att.view(-1,x.size(1),y.size(1))#[batch_size*doc_size*question_size]
        
        weighted_question = y_att.bmm(y)#[batch_size*doc_size*embedding_dim]
        return weighted_question
        
class LinearSeqAtt(nn.Module):
    """question self-aligened module"""
    def __init__(self,input_size):
        super(LinearSeqAtt,self).__init__()
        self.linear = nn.Linear(input_size,1)
        
    def forward(self,x,x_mask):
        '''
        input:
            x: (batch_size,sen_size,input_size)
            x_mask: (batch_size,sen_size)
        return:
            att: (batch_size,sen_size)
        '''
        x_flat = x.view(-1,x.size(-1))
        scores = self.linear(x_flat).view(x.size(0),x.size(1))#[batch_size*sen_size]
        scores.data.masked_fill_(x_mask.data,-float('inf'))
        att = F.softmax(scores)
        return att
        
class BilinearSeqAtt(nn.Module):
    """the output module"""
    def __init__(self,x_dim,y_dim):
        '''
        y_dim: the dim of doc feature vector
        x_dim: the dim of question feature vector
        '''
        super(BilinearSeqAtt,self).__init__()
        self.linear = nn.Linear(y_dim,x_dim)
    
    def forward(self,x,y,x_mask):
        '''
        input:
            x: (batch_size,xsen_size,x_feature_dim)
            y: (batch_size,y_feature_dim)
            x_mask: (batch_size,xsen_size)
        return:
            att: (batch_size,xsen_size)
            att_soft: (batch_size,xsen_size)
        '''
        Wy = self.linear(y) #[batch_size*doc_dim]
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data,-float('inf'))
        if self.training:
            att = F.log_softmax(xWy)
        else:
            att = F.softmax(xWy)
        att_soft = F.softmax(xWy)
        return att,att_soft
        
class AnswerModule(nn.Module):
    """the answer inference module(multi-hop)"""
    def __init__(self,x_dim,y_dim,hop_nums):
        '''
        x_dim: doc_feature_dim
        y_dim: question_feature_dim
        '''
        super(AnswerModule,self).__init__()
        self.slinear = nn.Linear(y_dim,x_dim)
        self.elinear = nn.Linear(y_dim,x_dim)       
        self.lstm = nn.LSTM(x_dim+y_dim,
                            y_dim,
                            num_layers=1,
                            batch_first=True)
        self.hop_nums = hop_nums
                            
    def forward(self,x,y,x_mask):
        '''
        input:
            x:(batch_size,xsen_size,x_dim)
            y:[batch_size,y_dim]
            x_mask:[batch_size,xsen_size]
        return:
            att: (batch_size,xsen_size)
            att_soft: (batch_size,xsen_size)            
        '''
        s = y.unsqueeze(1)
        h0 = s.transpose(0,1)
        c0 = s.transpose(0,1)
        hidden = (h0,c0)
        for i in range(self.hop_nums):
            Ws = self.slinear(s) #[batch_size,1,doc_dim]
            xWs = x.bmm(Ws.transpose(1,2)).squeeze(2) #[batch_size*doc_size]
            xWs.data.masked_fill_(x_mask.data,-float('inf')) 
            s_att = F.softmax(xWs) #[batch_size,doc_size]
            u = s_att.unsqueeze(1).bmm(x) #[batch_size,1,doc_dim]
            u = torch.cat([u,s],2)
            e,hidden = self.lstm(u,hidden) #[batch_size,1,que_dim]
            We = self.elinear(e)
            xWe = x.bmm(We.transpose(1,2)).squeeze(2)
            xWe.data.masked_fill_(x_mask.data,-float('inf'))
            if i < self.hop_nums - 1:
                e_att = F.softmax(xWe)
                u2 = e_att.unsqueeze(1).bmm(x)
                u2 = torch.cat([u2,e],2)
                s,hidden = self.lstm(u2,hidden)
        if self.training:
            s_att = F.log_softmax(xWs)
            e_att = F.log_softmax(xWe)
        else:
            s_att = F.softmax(xWs)
            e_att = F.softmax(xWe)
        s_att_soft = F.softmax(xWs)
        e_att_soft = F.softmax(xWe)
        att = [s_att,e_att]
        att_soft = [s_att_soft,e_att_soft]
        return att,att_soft

class SelfAttention(nn.Module):
    """self-attention module"""
    def __init__(self,input_size):
        super(SelfAttention,self).__init__()
        self.linear = nn.Linear(input_size,input_size)
        
    def forward(self,x,x_mask):
        '''
        input:
            x:(batch_size,xsen_size,x_feature_dim)
            x_mask:(batch_size,xsen_size)
        return:
            weighted_x: (batch_size,xsen_size,x_feature_dim)
        '''
        x_proj = self.linear(x.view(-1,x.size(2))).view(x.size())
        x_proj = F.relu(x_proj)
        
        scores = torch.bmm(x,x.transpose(1,2)) #[batch_size,doc_size,doc_size]
        x_mask = x_mask.unsqueeze(1).expand(scores.size()) #[batch_size,doc_size,doc_size]
        scores.data.masked_fill_(x_mask.data,-float('inf'))
        
        self_mask = Variable(torch.eye(x.size(1))).byte()
        if torch.cuda.is_available():
            self_mask = self_mask.cuda()
        self_mask = self_mask.unsqueeze(0).expand(scores.size())
        scores.data.masked_fill_(self_mask.data,-float('inf')) #[batch_size,doc_size,doc_size]
        
        att = F.softmax(scores.view(-1,x.size(1))) #[batch_size*doc_size,doc_size]
        att = att.view(x.size(0),x.size(1),x.size(1))
        
        weighted_x = att.bmm(x)#[batch_size*doc_size*encoding_dim]
        return weighted_x
        
class Seq2DocAtt(nn.Module):
    """question-to-doc representation（sentence level）"""
    def __init__(self,x_dim,y_dim):
        super(Seq2DocAtt,self).__init__()
        self.doc_linear = nn.Linear(x_dim,y_dim)
        self.seq_linear = nn.Linear(y_dim,y_dim)
        
    def forward(self,x,y,x_mask):
        '''
        input:
            x:(batch_size,xsen_size,x_feature_dim)
            y:(batch_size,y_feature_dim)
            x_mask:(batch_size,xsen_size)
        return:
            weighted_x: (batch_size,x_feature_dim)
        '''
        x_proj = self.doc_linear(x.view(-1,x.size(2))).view(x.size(0),x.size(1),y.size(1))
        x_proj = F.relu(x_proj)
        y_proj = self.seq_linear(y)
        y_proj = F.relu(y_proj)
        
        y_proj = y_proj.unsqueeze(1)
        scores = torch.bmm(y_proj,x_proj.transpose(1,2)) #[batch_size,1,doc_size]
        scores = scores.squeeze(1)
        scores.data.masked_fill_(x_mask.data,-float('inf'))
        att = F.softmax(scores) #[batch_size,doc_size]
        weighted_x = torch.bmm(att.unsqueeze(1),x).squeeze(1) #[batch_size,encoding_dim]
        return weighted_x
        
class MLPModule(nn.Module):
    def __init__(self,input_size,output_size,num_layers=1):
        super(MLPModule,self).__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList([nn.Linear(input_size,output_size) for _ in range(num_layers)])
    
    def forward(self,x):
        for layer in range(self.num_layers):
            x = self.linears[layer](x)
            x = F.relu(x)
        return x
        
                     

class CharEmbedding(nn.Module):
    """char-embedding encoding module"""
    def __init__(self,
                 vocab_size,
                 input_size,
                 out_channels,
                 kernel_size,
                 dropout_rate=0.4,
                 padding_idx=0):
        '''
        vocab_size: the size of char vocab
        input_size: the dim of char-embedding
        out_channels: the dim of char-embedding word encoding
        kernel_size: conv-kernel
        '''
        super(CharEmbedding,self).__init__()
        self.embeddings = nn.Embedding(vocab_size,input_size,padding_idx=padding_idx)
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList([nn.Conv2d(1,out_channels,(K,input_size),padding=(K-1,0))
                                   for K in kernel_size])
        
    def forward(self,x_in):
        '''
        input:
            x_in: (batch_size,sen_size,word_size)
        return:
            x: (batch_size,sen_size,2*out_channels) 
        '''
        x = x_in.view(-1,x_in.size(2))
        x = self.embeddings(x)
        if self.dropout_rate > 0:
            x = F.dropout(x,p=self.dropout_rate,training=self.training)
        x = x.unsqueeze(1)#x:(batch_size*sent_size)*1*word_size*char_embeddings
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]#[(batch_size*sent_size)*out_channels*H]
        x = [F.max_pool1d(i,i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x,1)
        x = x.view(x_in.size(0),x_in.size(1),-1)
        if self.dropout_rate > 0:
            x = F.dropout(x,p=self.dropout_rate,training=self.training)
        return x

class AttDecoder(nn.Module):
    """attention deocder（question generation module）"""
    def __init__(self,
                 vocab_size,
                 input_size,
                 hidden_size,
                 embedding,
                 dropout_rate,
                 qix_to_aix,
                 sos_idx):
        '''
        vocab_size: the size of question vocab
        input_size: the feature dim of the input encoding
        hidden_size: the dim of lstm hidden output 
        embedding: from the qa module
        dropout_rate: dropout rate
        qix_to_aix: the map of question dict to the whole dict
        '''
        super(AttDecoder,self).__init__()
        self.vocab_size = vocab_size        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sos_idx = sos_idx
        self.embedding = embedding
        self.dropout_rate = dropout_rate
        self.qix_to_aix = qix_to_aix
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            num_layers=1,
                            batch_first=True)
        self.out = nn.Linear(self.hidden_size,self.vocab_size)
        self.encoder2decoder = nn.Linear(self.hidden_size,self.hidden_size)
        self.att_linear = nn.Linear(self.hidden_size*3,self.hidden_size)
        
    def forward(self,input_h,q_att,max_len):
        '''
        input:
            input_h: (batch_size,doc_sen,doc_dim)
            q_att: (batch_size,doc_dim)
        return:
            encoder_outputs: (batch_size, doc_sen+1, vocab_size)
        '''
        q_att = q_att.unsqueeze(0)
        idx,h0,c0 = self.init_input(input_h) #batch_size * 1
        hidden = (q_att,c0)
        idx = self.ix_map(idx)
        encoder_list = []
        for i in range(max_len + 1):
            rnn_input = self.embedding(idx) #batch_size * 1 * embedding_size
            rnn_input = F.relu(rnn_input) #batch_size * 1 * input_size
            rnn_input = F.dropout(rnn_input,p=self.dropout_rate,training=self.training)
            output,hidden = self.lstm(rnn_input,hidden) #output:[batch_size*1*hidden_size] hidden:(h,c) [(layer)*batch_size*hidden]
            
            output_sim = self.out(output.squeeze(1)) #batch_size*vocab_size
            output_prob_log = F.log_softmax(output_sim).unsqueeze(1)  #batch_size*1*vocab_size
            encoder_list.append(output_prob_log)
            output_prob =F.softmax(output_sim) #batch_size*vocab_size
            idx = output_prob.max(1,keepdim=True)[1] #batch_size * 1
            idx = self.ix_map(idx)
            
        encoder_outputs = torch.cat(encoder_list,1) #batch_size * (sen_size+1) * vocab_size
        return encoder_outputs
        
    def init_input(self,h):
        result = torch.LongTensor([self.sos_idx]) #sos_idx
        result = result.expand(h.size(0)).unsqueeze(1) #batch_size * 1
        h0_encoder = torch.zeros(1,h.size(0),self.hidden_size)
        c0_encoder = torch.zeros(1,h.size(0),self.hidden_size)
        if torch.cuda.is_available():
            result = result.cuda()
            h0_encoder = h0_encoder.cuda()
            c0_encoder = c0_encoder.cuda()
        return Variable(result),Variable(h0_encoder),Variable(c0_encoder)
        
    def ix_map(self,qix):#batch_size*1
        qix = qix.squeeze(1)
        aix_list = []
        for ix in qix:
            aix = self.qix_to_aix[ix.data[0]]
            aix_list.append(aix)
        aix_tensor = torch.LongTensor(aix_list)
        if torch.cuda.is_available():
            aix_tensor = aix_tensor.cuda()
        aix_tensor = aix_tensor.unsqueeze(1) #batch_size*1
        return Variable(aix_tensor)
        
    def att_seq(self,xs,h): #xs[batch_size*sen_size*(3*hidden_dim)] h[batch_size*hidden_dim]
        xs_proj = self.att_linear(xs) #batch_size*sen_size*hidden_dim
        h_proj = h.unsqueeze(2) #batch*hidden_dim*1
        sim = xs_proj.bmm(h_proj) #batch_size*sen_size*1
        xs_att = xs.transpose(1,2).bmm(sim) #batch_size*(3*hidden_dim)*1
        xs_att = xs_att.transpose(1,2)
        return xs_att.contiguous() #batch_size*1*(3*hidden_dim)
        
        

def weighted_avg(x,x_weights):
    '''
    question merging function
    input:
        x: (batch_size,xsen_size,x_feature_dim)
        x_weights: (batch_size,xsen_size)
    return:
        (batch_size,x_feature_dim)
    '''
    return x_weights.unsqueeze(1).bmm(x).squeeze(1)
    
        
        