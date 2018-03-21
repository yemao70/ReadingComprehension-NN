# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 22:19:30 2017

@author: lcr
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 12 10:25:05 2017

@author: lcr
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import utils
import sys
import evaluate
import json
import os
from as_reader import ASReader
from as_reader import Discriminator
from optparse import OptionParser
import pickle
from torch.autograd import Variable
     
def get_paras():
    opt = OptionParser()
    opt.add_option('--rand_seed',
                   dest='rand_seed',
                   type=int,
                   default=22,
                   help='the random seed')    
    opt.add_option('--embedding_dim',
                   dest='embedding_dim',
                   type=int,
                   default=300,
                   help='the dim of word embedding')
    opt.add_option('--hidden_dim',
                   dest='hidden_dim',
                   type=int,
                   default=128,
                   help='the dim of rnn hidden layer')
    opt.add_option('--batch_size',
                   dest='batch_size',
                   type=int,
                   default=32,
                   help='thg size of the mini-batch')
    opt.add_option('--epochs',
                   dest='epochs',
                   type=int,
                   default=400,
                   help='the num of training epochs')
    opt.add_option('--pretrain_path',
                   dest='pretrain_path',
                   type=str,
                   default='/home/crli/crli/glove.840B.300d.txt',
                   help='the path of pre_train')
    opt.add_option('--train_path',
                   dest='train_path',
                   type=str,
                   default='./squad/train-v1.1.json',
                   help='the path of train data')
    opt.add_option('--dev_path',
                   dest='dev_path',
                   type=str,
                   default='./squad/dev-v1.1.json',
                   help='the path of dev data')
    opt.add_option('--dropout_rate',
                   dest='dropout_rate',
                   type=float,
                   default=0.3,
                   help='the dropout rate of rnn/embedding')
    opt.add_option('--word_dropout_use',
                   action='store_true',
                   dest='word_dropout_use',
                   default=False,
                   help='wheter to use word dropout'
                   )
    opt.add_option('--word_dropout_rate',
                   dest='word_dropout_rate',
                   type=float,
                   default=0.05,
                   help='the word dropout rate')
    opt.add_option('--rnn_layer_num',
                   dest='rnn_layer_num',
                   type=int,
                   default=3,
                   help='the num of rnn layer')
    opt.add_option('--fixed_embedding_num',
                   dest='fixed_embedding_num',
                   type=int,
                   default=1000,
                   help='the num of the fixed word embedding')
    opt.add_option('--max_span',
                   dest='max_span',
                   type=int,
                   default=15,
                   help='the max span in document during inference')
    opt.add_option('--extra_feature_num',
                   dest='extra_feature_num',
                   type=int,
                   default=4,
                   help='the num of extra features(lemma_match,uncase_match,exact_match,tf)')
    opt.add_option('--char_embedding_use',
                   action='store_true',
                   dest='char_embedding_use',
                   default=False,
                   help='whether to use char embedding')
    opt.add_option('--cove_embedding_use',
                   action='store_true',
                   dest='cove_embedding_use',
                   default=False,
                   help='whether to use cove embedding')
    opt.add_option('--cove_embedding_dim',
                   dest='cove_embedding_dim',
                   type=int,
                   default=300,
                   help='the dim of cove embedding')
    opt.add_option('--char_embedding_dim',
                   dest='char_embedding_dim',
                   type=int,
                   default=50,
                   help='the dim of char embedding')
    opt.add_option('--char_embedding_hidden_dim',
                   dest='char_embedding_hidden_dim',
                   type=int,
                   default=50,
                   help='the hidden dim of the char embedding out-channel')
    opt.add_option('--question_type_num',
                   dest='question_type_num',
                   type=int,
                   default=9,
                   help='the num of question type')
    opt.add_option('--question_type_dim',
                   dest='question_type_dim',
                   type=int,
                   default=10,
                   help='the embedding dim of the uestion_type')
    opt.add_option('--question_generate_use',
                   action='store_true',
                   dest='question_generate_use',
                   default=False,
                   help='whether to use question generate module')
    opt.add_option('--question_classification_use',
                   action='store_true',
                   dest='question_classification_use',
                   default=False,
                   help='whether to use question classification module')
#    opt.add_option('--kernel_sizes',
#                   dest='kernel_sizes',
#                   type=list,
#                   default=[2,3])
    opt.add_option('--pos_dim',
                   dest='pos_dim',
                   type=int,
                   default=20)
    opt.add_option('--ner_dim',
                   dest='ner_dim',
                   type=int,
                   default=20)
    opt.add_option('--test_only',
                   action='store_true',
                   dest='test_only',
                   default=False,
                   help='whether test only')
    opt.add_option('--model_path',
                   dest='model_path',
                   type=str,
                   default='.')
    opt.add_option('--save_model_use',
                    action='store_true',                   
                   dest='save_model_use',
                   default=False,
                   help='whether save model')
    opt.add_option('--first_load_data',
                   action='store_true',
                   dest='first_load_data',
                   default=False,
                   help='whether load data firstly')
    opt.add_option('--train_data',
                   dest='train_data',
                   type=str,
                   default='./data/train-v1.1')
    opt.add_option('--dev_data',
                   dest='dev_data',
                   type=str,
                   default='./data/dev-v1.1')
    opt.add_option('--generate_output_use',
                   action='store_true',
                   dest='generate_output_use',
                   default=False,
                   help='whether output the result of generated question')
    opt.add_option('--generate_output_file',
                   dest='generate_output_file',
                   default='./question_generate.out',
                   type=str,
                   help='generate file path')
    opt.add_option('--first_load_dict',
                   action='store_true',
                   dest='first_load_dict',
                   default=False,
                   help='whether load dict firstly')
    opt.add_option('--all_dict',
                   dest='all_dict',
                   default='dict/all_dict',
                   type=str,
                   help='the all_dict path')
    opt.add_option('--question_dict',
                   dest='question_dict',
                   default='dict/question_dict',
                   type=str,
                   help='the question_dict path')
    opt.add_option('--alpha',
                   dest='alpha',
                   default=0.8,
                   type=float,
                   help='the controller of loss')
    opt.add_option('--qa_train_time',
                   dest='qa_train_time',
                   default=-1,
                   type=int,
                   help='when to qa training')
    opt.add_option('--qg_train_time',
                   dest='qg_train_time',
                   default=-1,
                   type=int,
                   help='when to qg training')
    opt.add_option('--pre_train_model',
                   dest='pre_train_model',
                   default='.',
                   type=str)
    opt.add_option('--pre_train_use',
                   action='store_true',
                   dest='pre_train_use',
                   default=False,
                   help='whether use the pre_train model')
    opt.add_option('--mix_ratio',
                   dest='mix_ratio',
                   default=100,
                   type=int,
                   help='training QG per mix_ratio mini-batch')
    opt.add_option('--d_steps',
                   dest='d_steps',
                   default=1,
                   type=int,
                   help='the update number of discriminator')
    opt.add_option('--g_steps',
                   dest='g_steps',
                   default=1,
                   type=int,
                   help='the update number of generator')
    opt.add_option('--adversarial_training',
                   action='store_true',
                   dest='adversarial_training',
                   default=False,
                   help='whether using adversarial training')
    opt.add_option('--hop_nums',
                   dest='hop_nums',
                   default=3,
                   type=int,
                   help='the number of answer generation hop')
    opt.add_option('--D_stop_time',
                   dest='D_stop_time',
                   default=5,
                   type=int,
                   help='the time of discriminator stops')
                   
    (options,args) = opt.parse_args()
    return options

options = get_paras()
print(options)

EMBEDDING_DIM = 300
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 400
EVALUATION_INTERVAL = 1
GLOVE_PATH = '/home/crli/crli/glove.840B.300d.txt'
DROPOUT = 0.3
LAYER_NUM = 3
FIXED_EMBEDDING_NUM = 1000
SPAN_LENGTH = 15
KERNEL_SIZES=[2,3]


TRAIN_PATH = './squad/train-v1.1.json'
DEV_PATH = './squad/dev-v1.1.json'
path_to_predictions= './predict/predict_result'
path_to_dev = DEV_PATH

if options.first_load_data:
    train_exs,dev_exs = utils.load_task(TRAIN_PATH,DEV_PATH)
    with open(options.train_data,'wb') as file:
        pickle.dump(train_exs,file)
    with open(options.dev_data,'wb') as file:
        pickle.dump(dev_exs,file)
else:
    with open(options.train_data,'rb') as file:
        train_exs = pickle.load(file)
    with open(options.dev_data,'rb') as file:
        dev_exs = pickle.load(file)
            

print('The size of train_set:',len(train_exs))
print('The size of dev_set:',len(dev_exs))
TRAIN_SIZE = len(train_exs)
DEV_SIZE = len(dev_exs)

def load_glovedict(filepath):
    embedding_words = set()
    with open(filepath,"r",encoding="utf8") as f:
        for line in f:
            w = utils.normalize_text(line.rstrip().split(' ')[0])
            embedding_words.add(w)
    print('Num words in glove dict is',len(embedding_words))
    return embedding_words
embedding_words = load_glovedict(GLOVE_PATH)

pos_dict = {}
pos_dict["__NULL__"] = 0
ner_dict = {}
ner_dict["__NULL__"] = 0
def get_feature_dict(dataset,pos_dict,ner_dict):
    for ex in dataset:
        for w in ex['doc_pos']:
            if w not in pos_dict:
                pos_dict[w] = len(pos_dict)
        for w in ex['doc_ner']:
            if w not in ner_dict:
                ner_dict[w] = len(ner_dict)

get_feature_dict(train_exs,pos_dict,ner_dict)
print(pos_dict)
print(ner_dict)

POS_NUM = len(pos_dict)
NER_NUM = len(ner_dict)
print('The pos dict size',POS_NUM)
print('The ner dict size',NER_NUM)


def load_dict(words,dic):
    for w in words:
        if w not in embedding_words:
            continue
        if w not in dic:
            dic[w] = 1
        else:
            dic[w] += 1
            
if options.first_load_dict:
    word_dict = {}
    train_doc_words = [w for ex in train_exs for w in ex['doc_tokens']]
    train_question_words = [w for ex in train_exs for w in ex['question']['question_tokens']]
    train_answer_words = [w for ex in train_exs for w in ex['answer_tokens']]
    dev_doc_words = [w for ex in dev_exs for w in ex['doc_tokens']]
    dev_question_words = [w for ex in dev_exs for w in ex['question']['question_tokens']]
    load_dict(train_doc_words+train_question_words+train_answer_words+dev_doc_words+dev_question_words,word_dict)
    
    que_word_dict = {}
    load_dict(train_question_words,que_word_dict)
    with open(options.all_dict,'wb') as file:
        pickle.dump(word_dict,file)
    with open(options.question_dict,'wb') as file:
        pickle.dump(que_word_dict,file)    
else:
    with open(options.all_dict,'rb') as file:
        word_dict = pickle.load(file)
    with open(options.question_dict,'rb') as file:
        que_word_dict = pickle.load(file)    
           
word_dict["__NULL__"] = 100000003
word_dict["__SOS__"] = 100000002
word_dict["__EOS__"] = 100000001
word_dict["__UNK__"] = 100000000
word_dict = sorted(word_dict.items(),key=lambda x:x[1],reverse=True)

word_to_ix = {}
ix_to_word = {}
for w,_ in word_dict:
    ix_to_word[len(word_to_ix)] = w
    word_to_ix[w] = len(word_to_ix)

print('the size of vocab:',len(word_to_ix))
print('the size of ix_to_word',len(ix_to_word))

que_word_dict["__NULL__"] = 100000003
que_word_dict["__SOS__"] = 100000002
que_word_dict["__EOS__"] = 100000001
que_word_dict["__UNK__"] = 100000000
que_word_list = sorted(que_word_dict.items(),key=lambda x:x[1],reverse=True)

que_to_ix = {}
ix_to_que = {}
for w,n in que_word_list[:35000]:
    ix_to_que[len(que_to_ix)] = w
    que_to_ix[w] = len(que_to_ix)

print('the size of question vocab:', len(que_to_ix))

def get_qix_to_aix(que_to_ix,word_to_ix):
    '''
    function:
        the map of question dict to the whole dict
    input:
        que_to_ix: the question dict
        word_to_ix: the whole dict
    return:
        the map
    '''
    qix_to_aix = {}    
    for w in que_to_ix:
        qix = que_to_ix[w]
        aix = word_to_ix[w]
        qix_to_aix[qix] = aix
    return qix_to_aix
    
qix_to_aix = get_qix_to_aix(que_to_ix,word_to_ix)
#print(qix_to_aix)

char_dict = {}
for w,_ in word_dict:
    for c in w:
        if c not in char_dict:
            char_dict[c] = 1
        else:
            char_dict[c] += 1

char_to_ix = {}
char_to_ix["__NULL__"] = 0
char_to_ix["__UNK__"] = 1
for w in char_dict:
    for c in w:
        if c not in char_to_ix and char_dict[c] > 100:
            char_to_ix[c] = len(char_to_ix)
print('the size of char_to_ix',len(char_to_ix))
CHAR_VOCAB_SIZE = len(char_to_ix)
SOS_IDX = word_to_ix['__SOS__']

torch.manual_seed(options.rand_seed)
np.random.seed(options.rand_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(options.rand_seed)


def load_embeddings(vocab,filepath,embedding_size):
    '''
    input:
        vocab: the vocab of squad
        filepath: glove file path
        embedding_size: embedding dim
    return:
        embeddings: the whole embedding
        fixed_embeddings: the fixed embedding
    '''
    embeddings = np.random.normal(0.00,1.00,[len(vocab),embedding_size])
    count = 0
    with open(filepath,"r",encoding="utf8") as f:               
        for line in f:
            word = line.rstrip().split(' ')[0]
            word = utils.normalize_text(word)
            if(word in vocab):
                count += 1
                vec = line.strip().split(" ")[1:]
                vec = np.array(vec)
                embeddings[vocab[word]] = vec
    embeddings[vocab["__NULL__"]] = np.zeros(embedding_size)
    fixed_embeddings = embeddings[FIXED_EMBEDDING_NUM+3:]
    print('the glove count:',count)
    return embeddings,fixed_embeddings
    
def sequence_label(batch_size_sen,to_ix):
    '''
    function:
        preparing the question label for qg loss
    '''
    seqs = []
    for s in batch_size_sen:
        s_inx = []
        for i,w in enumerate(s):
            if w in to_ix:
                s_inx.append(to_ix[w])
            else:
                s_inx.append(to_ix['__UNK__'])
        s_inx.append(to_ix['__EOS__'])
        seqs.append(s_inx)
    lengths = [len(seq) for seq in seqs]
    n_examples = len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_examples,max_len)).astype('int64')
    for idx,seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
    x_tensor = torch.LongTensor(x)
    if torch.cuda.is_available():
        x_tensor = x_tensor.cuda()
    x_var = autograd.Variable(x_tensor)
    return x_var
    

def prepare_sentence(batch_size_sen,to_ix,char_to_ix):
    '''
    function:
        transfer the sentence to the index reoresentation of vocab
    '''
    seqs = []
    cseqs = []
    for s in batch_size_sen:
        s_inx = []
        c_inxs = []
        for i,w in enumerate(s):
            c_inx = []
            if w in to_ix:
                s_inx.append(to_ix[w])
            else:
                s_inx.append(to_ix['__UNK__'])
#            if i == len(s) -1 :
#                s_inx.append(to_ix['__EOS__'])
            for c in w:
                if c in char_to_ix:
                    c_inx.append(char_to_ix[c])
                else:
                    c_inx.append(char_to_ix['__UNK__'])
            c_inxs.append(c_inx)
        seqs.append(s_inx)
        cseqs.append(c_inxs)
    lengths = [len(seq) for seq in seqs]
    n_examples = len(seqs)
    max_len = np.max(lengths)
    max_word_len = np.max([len(word) for cseq in cseqs for word in cseq])
    x = np.zeros((n_examples,max_len)).astype('int64')
    x_mask = np.ones((n_examples,max_len)).astype('uint8')
    c = np.zeros((n_examples,max_len,max_word_len)).astype('int64')
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
        x_mask[idx, :lengths[idx]] = 0
    for idx, seq in enumerate(cseqs):
        for idx2, word in enumerate(seq):
            c[idx, idx2, :len(word)] = word
    x_tensor = torch.LongTensor(x)
    c_tensor = torch.LongTensor(c)
    x_mask_tensor = torch.ByteTensor(x_mask)    
    if torch.cuda.is_available():
        x_tensor = x_tensor.cuda()
        c_tensor = c_tensor.cuda()
        x_mask_tensor = x_mask_tensor.cuda()
    return autograd.Variable(x_tensor,requires_grad=False),autograd.Variable(c_tensor,requires_grad=False),autograd.Variable(x_mask_tensor,requires_grad=False)
    
def prepare_feature(batch_features,batch_pos,batch_ner):
    '''
    function:
        preparing thee feature(pos, ner, tf) vector
    '''
    lengths = [len(sen) for sen in batch_features]
    max_len = np.max(lengths)
    n_examples = len(batch_features)
    x = np.zeros((n_examples,max_len,options.extra_feature_num))       
    for idx,f in enumerate(batch_features):
        x[idx, :lengths[idx]] = f
    x_tensor = torch.Tensor(x)
    poss_list = [] 
    for poss in batch_pos:
        pos_list = []
        for pos in poss:
            if pos not in pos_dict:
                index = pos_dict["__NULL__"]
            else:
                index = pos_dict[pos]
            pos_list.append(index)
        poss_list.append(pos_list)
    ners_list = []
    for ners in batch_ner:
        ner_list = []
        for ner in ners:
            if ner not in ner_dict:
                index = ner_dict["__NULL__"]
            else:
                index = ner_dict[ner]
            ner_list.append(index)
        ners_list.append(ner_list)
    y = np.zeros((n_examples,max_len)).astype('int64')
    for idx,pos in enumerate(poss_list):
        y[idx, :lengths[idx]] = pos
    y_tensor = torch.LongTensor(y)
    z = np.zeros((n_examples,max_len)).astype('int64')
    for idx,ner in enumerate(ners_list):
        z[idx, :lengths[idx]] = ner
    z_tensor = torch.LongTensor(z)
    
    if torch.cuda.is_available():
        x_tensor = x_tensor.cuda()
        y_tensor = y_tensor.cuda()
        z_tensor = z_tensor.cuda()
    f_in = [autograd.Variable(x_tensor,requires_grad=False),
            autograd.Variable(y_tensor,requires_grad=False),
            autograd.Variable(z_tensor,requires_grad=False)]
    return f_in

def prepare_target(batch_label):
    '''
    function:
        preparing the correct answer label for qa loss
    '''
    target_start = [x[0] for x in batch_label]
    target_end = [x[1] for x in batch_label]
    tensor_start = torch.LongTensor(target_start)
    tensor_end = torch.LongTensor(target_end)
    if torch.cuda.is_available():
        tensor_start = tensor_start.cuda()
        tensor_end = tensor_end.cuda()
    return autograd.Variable(tensor_start,requires_grad=False),autograd.Variable(tensor_end,requires_grad=False)                       
    
def get_predict_text(start_scores,end_scores,c_batch,spans):
    '''
    function:
       transfer the start and end index to text
    '''
    start_scores = start_scores.cpu().data
    end_scores = end_scores.cpu().data
    answer=[]
    max_len = SPAN_LENGTH or start_scores.size(1)
    for i in range(start_scores.size(0)):
        scores = torch.ger(start_scores[i],end_scores[i])
        scores.triu_().tril_(max_len - 1)
        scores = scores.numpy()
        s_idx,e_idx = np.unravel_index(np.argmax(scores),scores.shape)
        s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
        a = c_batch[i][s_offset:e_offset]
        answer.append(a)
    return answer   
    
def question_type_prepare(question_type):
    '''
    function:
        transfer the question type to tensor
    '''
    question_type_tensor = torch.LongTensor(question_type)
    if torch.cuda.is_available():
        question_type_tensor = question_type_tensor.cuda()
    return autograd.Variable(question_type_tensor,requires_grad=False)
    
def evaluation_devresult(pre_result,target_result):
    '''
    function:
        QA evaluation
    '''
    f1 = exact_match = total = 0
    for i in range(len(pre_result)):
        total += 1
        prediction = pre_result[i]
        ground_truths = target_result[i]
        exact_match += evaluate.metric_max_over_ground_truths(evaluate.exact_match_score, prediction, ground_truths)
        f1 += evaluate.metric_max_over_ground_truths(evaluate.f1_score, prediction, ground_truths)
    exact_match =  exact_match / total
    f1 =  f1 / total
    return exact_match,f1

def predict_dump(q_ids,predict):
    result = {}
    for i in range(len(q_ids)):
        result[q_ids[i]] = predict[i]
    f = open(path_to_predictions,"w",encoding="utf-8")
    print(json.dumps(result),file=f)
    f.close()
    
def question_restore(scores):#scores[batch_size*sentence_size*vocab_size]
    scores = scores.max(2)[1] #[batch_size*sentence_size]
    question_predicts = []
    for i in range(len(scores)):
        question_predict = []
        for idx in scores[i]:
            question_predict.append(ix_to_que[idx.data[0]])
        question_predicts.append(question_predict)
#        print(question_predict)
#        print(source[i])
#        print('----------------')
    return question_predicts

def evaluate_generate_accruacy(predict,target):
    '''
    function:
        QG evaluation 
    '''
    correct_num = 0
    sum_ = 0
    for i in range(len(predict)):
        if int(target[i].data[0] == 0):
            continue
        sum_ += 1
        if int(predict[i].data[0]) == int(target[i].data[0]):
            correct_num += 1
    return correct_num,sum_
    
def generate_question_print(generate_question,original_question):
    f = open(options.generate_output_file,"a",encoding='utf-8')
    for i in range(len(generate_question)):
        print(generate_question[i],file=f)
        print(original_question[i],file=f)
        print('---------------',file=f)
    f.close()

def get_answer_idx(label):
    s_idx = [l[0] for l in label]
    e_idx = [l[1] for l in label]
    answer_idx = [s_idx,e_idx]
    return answer_idx
    
def get_dev_answer_idx(label):
    s_idx = [ls[0][0] for ls in label]
    e_idx = [ls[0][1] for ls in label]
    answer_idx = [s_idx,e_idx]
    return answer_idx
        
    
best_accuracy = 0
best_f1 = 0
best_generate = 0

#def model_predict(model):
#    model.eval()
#    pre_result = []
#    target_result = []
#    q_ids = []
#    correct_ = 0
#    sum_ = 0
#    loss_qg = 0
#
#    for start in range(0,DEV_SIZE,BATCH_SIZE):
#        end = start + BATCH_SIZE
#        c = [ex['doc_text'] for ex in dev_exs[start:end]]
#        s = [ex['doc_tokens'] for ex in dev_exs[start:end]]
#        pos = [ex['doc_pos'] for ex in dev_exs[start:end]]
#        ner = [ex['doc_ner'] for ex in dev_exs[start:end]]
#        q = [ex['question']['question_tokens'] for ex in dev_exs[start:end]]
#        q_id = [ex['question']['question_id'] for ex in dev_exs[start:end]]
#        q_t = [ex['question']['question_type'] for ex in dev_exs[start:end]]
#        q_t = question_type_prepare(q_t)
#        q_seq_label = sequence_label(q,que_to_ix)
#        a = [ex['answer_text'] for ex in dev_exs[start:end]]
#        l = [ex['label'] for ex in dev_exs[start:end]]
#        f = [ex['doc_features'] for ex in dev_exs[start:end]]
#        spans = [ex['doc_spans'] for ex in dev_exs[start:end]]
#        answer_idx = get_dev_answer_idx(l)
#        s_in,sc_in,s_mask = prepare_sentence(s,word_to_ix,char_to_ix)
#        q_in,qc_in,q_mask = prepare_sentence(q,word_to_ix,char_to_ix)
#        f_in = prepare_feature(f,pos,ner)
#        if options.question_generate_use:
#            start_score,end_score,seq_scores = model(s_in,sc_in,f_in,s_mask,q_in,qc_in,q_t,q_mask,answer_idx)
#            loss = F.nll_loss(seq_scores,q_seq_label,weight=weight_mask)
#            loss_qg += loss.cpu().data.numpy()
#            if options.generate_output_use:                
#                generate_question = question_restore(seq_scores.view(q_in.size(0),-1,len(que_to_ix)))
#                generate_question_print(generate_question,q)
#            c_,s_ = evaluate_generate_accruacy(seq_scores.max(1)[1],q_seq_label)
#            correct_ += c_
#            sum_ += s_
#        else:
#            start_score,end_score = model(s_in,sc_in,f_in,s_mask,q_in,qc_in,q_t,q_mask,answer_idx)
#        pre_answer_batch = get_predict_text(start_score,end_score,c,spans)
#        pre_result.extend(pre_answer_batch)
#        target_result.extend(a)
#        q_ids.extend(q_id)        
#    em,f1 = evaluation_devresult(pre_result,target_result)
#    global best_accuracy
#    global best_f1
#    global best_generate
#    if(em > best_accuracy):
#        if(options.save_model_use):
#            torch.save(model.state_dict(),options.model_path)
#        best_accuracy = em
#    if(f1 > best_f1):
#        best_f1 = f1
#    if options.question_generate_use:
#        generata_acc = correct_/sum_
#        if generata_acc > best_generate:
#            if(options.save_model_use):
#                torch.save(model.state_dict(),options.model_path+'_QG')
#            best_generate = generata_acc
#        print('dev dataset {EM:%.5f, F1:%.5f, Best_EM:%.5f, Best_F1:%.5f, GenerateAcc:%.5f, QG Loss:%.2f, Best_GenerateAcc:%.5f}' %(em,f1,best_accuracy,best_f1,generata_acc,loss_qg,best_generate))
#    else:
#        print('dev dataset {EM:%.5f, F1:%.5f, Best_EM:%.5f, Best_F1:%.5f}' %(em,f1,best_accuracy,best_f1))
#    print('------------------------------')

def model_predict(model):
    model.eval()
    pre_result = []
    target_result = []
    q_ids = []   
    
    for start in range(0,DEV_SIZE,BATCH_SIZE):
        end = start + BATCH_SIZE
        c = [ex['doc_text'] for ex in dev_exs[start:end]]
        s = [ex['doc_tokens'] for ex in dev_exs[start:end]]
        pos = [ex['doc_pos'] for ex in dev_exs[start:end]]
        ner = [ex['doc_ner'] for ex in dev_exs[start:end]]
        q = [ex['question']['question_tokens'] for ex in dev_exs[start:end]]
        q_id = [ex['question']['question_id'] for ex in dev_exs[start:end]]
        q_t = [ex['question']['question_type'] for ex in dev_exs[start:end]]
        q_t = question_type_prepare(q_t)
        a = [ex['answer_text'] for ex in dev_exs[start:end]]
        l = [ex['label'] for ex in dev_exs[start:end]]
        f = [ex['doc_features'] for ex in dev_exs[start:end]]
        spans = [ex['doc_spans'] for ex in dev_exs[start:end]]
        answer_idx = get_dev_answer_idx(l)
        s_in,sc_in,s_mask = prepare_sentence(s,word_to_ix,char_to_ix)
        q_in,qc_in,q_mask = prepare_sentence(q,word_to_ix,char_to_ix)
        f_in = prepare_feature(f,pos,ner)

        start_score,end_score = model(s_in,sc_in,f_in,s_mask,q_in,qc_in,q_t,q_mask,answer_idx)
        pre_answer_batch = get_predict_text(start_score,end_score,c,spans)
        pre_result.extend(pre_answer_batch)
        target_result.extend(a)
        q_ids.extend(q_id)        
    em,f1 = evaluation_devresult(pre_result,target_result)
    global best_accuracy
    global best_f1
    if(em > best_accuracy):
        if(options.save_model_use):
            torch.save(model.state_dict(),options.model_path)
        best_accuracy = em
    if(f1 > best_f1):
        best_f1 = f1

    print('dev dataset {EM:%.5f, F1:%.5f, Best_EM:%.5f, Best_F1:%.5f}' %(em,f1,best_accuracy,best_f1))
    print('------------------------------')       
           

init_embeddings,fixed_embeddings = load_embeddings(word_to_ix,
                                                   options.pretrain_path,
                                                   options.embedding_dim)

model = ASReader(options,
                 POS_NUM,
                 NER_NUM,
                 CHAR_VOCAB_SIZE,
                 KERNEL_SIZES,
                 len(word_to_ix),
                 len(que_to_ix),
                 torch.FloatTensor(init_embeddings),
                 torch.FloatTensor(fixed_embeddings),
                 qix_to_aix,
                 SOS_IDX)
D_model = Discriminator(options,
                        model.doc_hidden_dim,
                        len(word_to_ix),
                        len(que_to_ix),
                        qix_to_aix,
                        SOS_IDX,
                        model.embedding)

        
alpha = options.alpha
weight_mask = torch.ones(len(que_to_ix))                 
if torch.cuda.is_available():
    model.cuda()
    D_model.cuda()
    weight_mask = weight_mask.cuda()
weight_mask[que_to_ix['__NULL__']] = 0

if options.pre_train_use:
    pre_train_state_dict = torch.load(options.pre_train_model)
    cur_state_dict = model.state_dict()
    for k, value in pre_train_state_dict.items():
        if 'seq_decoder_rnn' not in k and k in cur_state_dict:
            cur_state_dict[k] = value
    model.load_state_dict(cur_state_dict)
    
def get_reward(score_soft,seq_scores,q_seq_label):
    '''
    start_score:(batch_size,doc_size)
    end_score:(batch_size,doc_size)
    seq_scores:(batch_size,sen_size,vocab_size)
    q_seq_label:(batch_size,sen_size)
    '''
    start_score = score_soft[0].max(1)[0] #[batch_size]
    end_score = score_soft[1].max(1)[0] #[batch_size]
    loss = Variable(torch.zeros(1))
    if torch.cuda.is_available():
        loss = loss.cuda()
    for i in range(seq_scores.size(0)):
        reward = F.nll_loss(seq_scores[i],q_seq_label[i],weight=weight_mask)
        loss += (start_score[i] * end_score[i]) * reward
    return loss

def get_sen_loss(seq_scores,q_seq_label):
    '''
    seq_scores:(batch_size,sen_size,vocab_size)
    q_seq_label:(batch_size,sen_size)
    '''    
    loss = Variable(torch.zeros(1))
    if torch.cuda.is_available():
        loss = loss.cuda()
    for i in range(seq_scores.size(0)):
        loss += F.nll_loss(seq_scores[i],q_seq_label[i],weight=weight_mask)
    return loss
    
def get_fake_sen_loss(seq_scores,q_seq_label,predict,target):
    '''
    seq_scores:(batch_size,sen_size,vocab_size)
    q_seq_label:(batch_size,sen_size)
    start_score:(batch_size,doc_size)
    end_score:(batch_size,doc_size)
    '''    
    loss = Variable(torch.zeros(1))
    start_predict = predict[0].max(1)[1] #[batch_size]
    end_predict = predict[1].max(1)[1] #[batch_size]
    start_target = target[0]
    end_target = target[1]
    if torch.cuda.is_available():
        loss = loss.cuda()
    for i in range(seq_scores.size(0)):
        if start_predict[i].data[0] != start_target[i].data[0] or end_predict[i].data[0] != end_target[i].data[0]: 
            loss += F.nll_loss(seq_scores[i],q_seq_label[i],weight=weight_mask)
    return loss
       

#def adversarial_train():
#    generator_parameters = [e for e in model.parameters() if e.requires_grad]
#    discriminator_parameters = [e for e in D_model.parameters() if e.requires_grad]
#    generator_optimizer = optim.Adamax(generator_parameters,weight_decay=0)
#    discriminator_optimizer = optim.Adamax(discriminator_parameters,weight_decay=0)
#    batches = [(start,start + BATCH_SIZE) for start in range(0,TRAIN_SIZE,BATCH_SIZE)]
#    iterator_nums = 1
#    loss_sum = 0
#    
#    for t in range(1,options.epochs+1):
#        np.random.shuffle(batches)
#        D_fake_loss = 0
#        for d_index in range(options.d_steps):
#            if t < options.D_stop_time:
#                for start,end in batches:
#                    D_model.train()
#                    model.train()
#                    if (end > len(train_exs)): end = len(train_exs)
#                    D_model.zero_grad()
#                    s = [ex['doc_tokens'] for ex in train_exs[start:end]]
#                    pos = [ex['doc_pos'] for ex in train_exs[start:end]]
#                    ner = [ex['doc_ner'] for ex in train_exs[start:end]]
#                    q = [ex['question']['question_tokens'] for ex in train_exs[start:end]]
#                    q_t = [ex['question']['question_type'] for ex in train_exs[start:end]]
#                    q_t = question_type_prepare(q_t)
#                    l = [ex['label'] for ex in train_exs[start:end]]
#                    f = [ex['doc_features'] for ex in train_exs[start:end]]
#                    s_in,sc_in,s_mask = prepare_sentence(s,word_to_ix,char_to_ix)
#                    q_in,qc_in,q_mask = prepare_sentence(q,word_to_ix,char_to_ix)
#                    answer_idx = get_answer_idx(l)
#                    q_seq_label = sequence_label(q,que_to_ix)
#                    f_in = prepare_feature(f,pos,ner)
#                    start_target,end_target = prepare_target(l)
#                    start_score,end_score,doc_encoding,score_soft = model(s_in,sc_in,f_in,s_mask,q_in,qc_in,q_t,q_mask,answer_idx)
#                    doc_encoding = doc_encoding.detach()
#                    score_soft[0] = score_soft[0].detach()
#                    score_soft[1] = score_soft[1].detach()
#                    seq_scores,seq_scores_gold = D_model(doc_encoding,score_soft,answer_idx,q_in.size(1))
#    #                real_data_loss = F.nll_loss(seq_scores_gold,q_seq_label,weight=weight_mask)
#                    real_data_loss = get_sen_loss(seq_scores_gold,q_seq_label)
#    #                fake_data_loss = -F.nll_loss(seq_scores,q_seq_label,weight=weight_mask)
#                    fake_data_loss = -get_fake_sen_loss(seq_scores,q_seq_label,score_soft,(start_target,end_target))
#                    D_loss = real_data_loss + fake_data_loss
#                    D_fake_loss += fake_data_loss.cpu().data.numpy()
#                    D_loss.backward()
#                    nn.utils.clip_grad_norm(discriminator_parameters,10)
#                    discriminator_optimizer.step()
#
#        G_fake_loss = 0
#        for g_index in range(options.g_steps):
#            for start,end in batches:
#                D_model.train()
#                model.train()
#                if (end > len(train_exs)): end = len(train_exs)
#                model.zero_grad()
#                s = [ex['doc_tokens'] for ex in train_exs[start:end]]
#                pos = [ex['doc_pos'] for ex in train_exs[start:end]]
#                ner = [ex['doc_ner'] for ex in train_exs[start:end]]
#                q = [ex['question']['question_tokens'] for ex in train_exs[start:end]]
#                q_t = [ex['question']['question_type'] for ex in train_exs[start:end]]
#                q_t = question_type_prepare(q_t)
#                l = [ex['label'] for ex in train_exs[start:end]]
#                f = [ex['doc_features'] for ex in train_exs[start:end]]
#                s_in,sc_in,s_mask = prepare_sentence(s,word_to_ix,char_to_ix)
#                q_in,qc_in,q_mask = prepare_sentence(q,word_to_ix,char_to_ix)
#                answer_idx = get_answer_idx(l)
#                q_seq_label = sequence_label(q,que_to_ix)
#                f_in = prepare_feature(f,pos,ner)
#                start_score,end_score,doc_encoding,score_soft = model(s_in,sc_in,f_in,s_mask,q_in,qc_in,q_t,q_mask,answer_idx)
#                seq_scores,seq_scores_gold = D_model(doc_encoding,score_soft,answer_idx,q_in.size(1))
#                start_target,end_target = prepare_target(l)
#                MLE = F.nll_loss(start_score,start_target)+F.nll_loss(end_score,end_target)
#                generate_data_reward = get_reward(score_soft,seq_scores,q_seq_label)
#                G_loss = generate_data_reward + MLE
#                G_fake_loss += generate_data_reward.cpu().data.numpy()
#                loss_sum += G_loss.cpu().data.numpy()
#                G_loss.backward()
#                nn.utils.clip_grad_norm(generator_parameters,10)
#                generator_optimizer.step()
#                model.reset_embeddings()
#                D_model.reset_embedding(model.embedding)
#
#                if iterator_nums % 10 == 0:
#                    print('Epochs: %d MiniBatch: %d Loss: %.2f' % (t,iterator_nums,loss_sum))
#                    loss_sum = 0
#                    if t > 0:
#                        model_predict(model)
#                iterator_nums += 1
                            
def train():
    #loss_function = nn.CrossEntropyLoss()
    #loss_function = nn.NLLLoss() 
    parameters = [e for e in model.parameters() if e.requires_grad]
    #optimizer = optim.SGD(parameters,lr=0.1)
    optimizer = optim.Adamax(parameters,weight_decay=0)
    #optimizer = optim.Adadelta(parameters,lr=0.5,weight_decay=1e-4)
    batches = [(start,start + BATCH_SIZE)for start in range(0,TRAIN_SIZE,BATCH_SIZE)]
        
    iterator_nums = 1
    loss_sum = 0
    loss_qa = 0
    loss_qg = 0
    
    for t in range(1,options.epochs+1):
    #    np.random.shuffle(train_exs)
        np.random.shuffle(batches)
        for start,end in batches:
            model.train()
            if(end > len(train_exs)): end = len(train_exs)
            model.zero_grad()
            s = [ex['doc_tokens'] for ex in train_exs[start:end]]
            pos = [ex['doc_pos'] for ex in train_exs[start:end]]
            ner = [ex['doc_ner'] for ex in train_exs[start:end]]
            q = [ex['question']['question_tokens'] for ex in train_exs[start:end]]
            q_t = [ex['question']['question_type'] for ex in train_exs[start:end]]
            q_t = question_type_prepare(q_t)
            a = [ex['answer_text'] for ex in train_exs[start:end]]
            l = [ex['label'] for ex in train_exs[start:end]]
            f = [ex['doc_features'] for ex in train_exs[start:end]]
            s_in,sc_in,s_mask = prepare_sentence(s,word_to_ix,char_to_ix)
            q_in,qc_in,q_mask = prepare_sentence(q,word_to_ix,char_to_ix)
            answer_idx = get_answer_idx(l)
            q_seq_label = sequence_label(q,que_to_ix)
            f_in = prepare_feature(f,pos,ner)
            if options.question_generate_use:
                start_score,end_score,seq_scores = model(s_in,sc_in,f_in,s_mask,q_in,qc_in,q_t,q_mask,answer_idx)
#                question_restore(seq_scores.view(q_in.size(0),-1,len(que_to_ix)),q)
                start_target,end_target = prepare_target(l)
#                loss1 = F.nll_loss(start_score,start_target)+F.nll_loss(end_score,end_target)
#                loss2 = F.nll_loss(seq_scores,q_seq_label)
#                print('QA loss:',loss1)
#                print('QG loss:',loss2)
                loss1 = F.nll_loss(start_score,start_target)+F.nll_loss(end_score,end_target)
                loss2 = F.nll_loss(seq_scores,q_seq_label,weight=weight_mask)
                loss = torch.Tensor([0])
                if torch.cuda.is_available():
                    loss = loss.cuda()
                loss = autograd.Variable(loss)
                if t > options.qa_train_time:
                    loss += loss1
                if t > options.qg_train_time:
                    loss += loss2
#                loss = alpha*loss1 + (1-alpha)*loss2
                loss_qa += loss1.cpu().data.numpy()
                loss_qg += loss2.cpu().data.numpy()
            elif options.question_classification_use:
                start_score,end_score,type_scores = model(s_in,sc_in,f_in,s_mask,q_in,qc_in,q_t,q_mask,answer_idx)
                start_target,end_target = prepare_target(l)
                loss = F.nll_loss(start_score,start_target)+F.nll_loss(end_score,end_target)+F.nll_loss(type_scores,q_t)
            else:
                start_score,end_score = model(s_in,sc_in,f_in,s_mask,q_in,qc_in,q_t,q_mask,answer_idx)
                start_target,end_target = prepare_target(l)
                loss = F.nll_loss(start_score,start_target)+F.nll_loss(end_score,end_target)
            loss_sum += loss.cpu().data.numpy()
            loss.backward()
            nn.utils.clip_grad_norm(parameters,10)
            optimizer.step()
            model.reset_embeddings()
            if iterator_nums % 10 == 0:
                if options.question_generate_use:
                    print('Epochs: %d MiniBatch: %d QA Loss: %.2f QG Loss: %.2f Loss: %.2f' % (t,iterator_nums,loss_qa,loss_qg,loss_sum))
                    loss_qa = 0
                    loss_qg = 0
                    loss_sum = 0 
                else:
                    print('Epochs: %d MiniBatch: %d Loss: %.2f' % (t,iterator_nums,loss_sum))
                    loss_sum = 0            
                if t > 0:
                    model_predict(model)
            iterator_nums += 1


if options.test_only:
#    if torch.cuda.is_available():
#        model.cuda()
#    model.load_state_dict(torch.load(options.model_path))

    pre_train_state_dict = torch.load(options.model_path)
    cur_state_dict = model.state_dict()
    for k, value in pre_train_state_dict.items():
        if k in cur_state_dict:
            cur_state_dict[k] = value
    model.load_state_dict(cur_state_dict)
    model_predict(model)
else:
    train()