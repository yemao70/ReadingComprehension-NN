# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 12:04:48 2017

@author: lcr
"""
import re
import json
import nltk
import sys
import spacy
from collections import Counter
import unicodedata
from stanfordcorenlp import StanfordCoreNLP
import spacy.en
import numpy as np

#stanforNLP = StanfordCoreNLP(r'/home/crli/stanford-corenlp-full-2017-06-09')


NLP = spacy.load('en')
#NLP = spacy.en.English()

def load_task(train_filepath,dev_filepath):
    print('loading dataset...')
    train_data = load_traindata(train_filepath)
    dev_data = load_devdata(dev_filepath)
    train_exs = vectorize_train(train_data)
    dev_exs = vectorize_dev(dev_data)
    print('loading over...')
    return train_exs,dev_exs


def spacyTokenzie(sent):
    tokens = NLP.tokenizer(sent)
    return [t.text for t in tokens]
    
#def standfordTokenize(sent):
#    tokens = stanforNLP.word_tokenize(sent)
#    return [token.replace("''", '"').replace("``", '"') for token in tokens]
    
def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def tokenize(sent):
    sens = nltk.sent_tokenize(sent)
    words = []
    for sen in sens:
        sen = nltk.word_tokenize(sen)
        for i in range(len(sen)):
            sen[i] = sen[i].replace("''", '\"')
            sen[i] = sen[i].replace("``", '\"')
        words.extend(sen)
    return words

nlp=StanfordCoreNLP(r'/home/crli/stanford-corenlp-full-2017-06-09',memory='8g')
   
def stanfordparser(sent):
    output = nlp.annotate(sent,properties={'annotators':'tokenize,ssplit,lemma,ner'})
    try:
        data = json.loads(output,strict=False)
    except:
        print(output)
        print(sent)
    tokens = []
    lemmas = []
    spans = []
    poss = []
    ners = []    
    for sid,sen in enumerate(data["sentences"]):
        for tid,token in enumerate(sen["tokens"]):
            tokens.append(token["originalText"])
            lemmas.append(token["lemma"])
            poss.append(token["pos"])
            ners.append(token["ner"])
            spans.append([token["characterOffsetBegin"],token["characterOffsetEnd"]])
    return tokens,lemmas,poss,ners,spans
    
def get_question_type(question_lemma):
    for w in question_lemma[0:4]:
        if w.lower() == "what":
            return 0
        elif w.lower() == "how":
            return 1
        elif w.lower() == "who":
            return 2
        elif w.lower() == "when":
            return 3
        elif w.lower() == "which":
            return 4
        elif w.lower() == "where":
            return 5
        elif w.lower() == "why":
            return 6
        elif w.lower() == "be":
            return 7
    return 8

def load_traindata(filepath):
    data = []
    with open(filepath,"r") as f:  
        source_data = json.load(f)
        for ai, article in enumerate(source_data['data'][0:1]):
            for pi, para in enumerate(article['paragraphs']):
                context = para['context']
#                context = context.replace("''", '\"')
#                context = context.replace("``", '\"')
                context_tokens,context_lemmas,context_poss,context_ners,context_spans = stanfordparser(context)
                doc = {}
                doc['context'] = context
                doc['context_tokens'] = context_tokens
                doc['context_lemmas'] = context_lemmas
                doc['context_poss'] = context_poss
                doc['context_ners'] = context_ners
                doc['context_spans'] = context_spans
#                doc = [context,context_tokens,context_lemmas,context_poss,context_ners,context_spans]
                for qa in para['qas']:
                    question_text = qa["question"].strip()
                    question_id = qa["id"].strip()
                    question_tokens,question_lemmas,_,_,_ = stanfordparser(question_text)
                    question_type = get_question_type(question_lemmas)
                    question = {}
                    question['question_id'] = question_id
                    question['question_tokens'] = question_tokens
                    question['question_lemmas'] = question_lemmas
                    question['question_type'] = question_type
#                    question = [question_id,question_tokens,question_lemmas]
                    for answer in qa['answers']:
                        answer_text = answer['text']
                        answer_tokens,_,_,_,_ = stanfordparser(answer_text)
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer = {}
                        answer['answer_text'] = answer_text
                        answer['answer_tokens'] = answer_tokens
                        answer['answer_idxs'] = (answer_start,answer_end)
#                        answer = [answer_text,answer_tokens,(answer_start,answer_end)]
                        data.append([doc,question,answer,len(context_tokens)])
    return data
    
def load_devdata(filepath):
    data = []
    with open(filepath,"r") as f:  
        source_data = json.load(f)
        for ai, article in enumerate(source_data['data'][0:1]):
            for pi, para in enumerate(article['paragraphs']):
                context = para['context']
#                context = context.replace("''", '\"')
#                context = context.replace("``", '\"')
                context_tokens,context_lemmas,context_poss,context_ners,context_spans = stanfordparser(context)
                doc = {}
                doc['context'] = context
                doc['context_tokens'] = context_tokens
                doc['context_lemmas'] = context_lemmas
                doc['context_poss'] = context_poss
                doc['context_ners'] = context_ners
                doc['context_spans'] = context_spans                
#                doc = [context,context_tokens,context_lemmas,context_poss,context_ners,context_spans]
                for qa in para['qas']:
                    question_text = qa["question"].strip()
                    question_id = qa["id"].strip()
                    question_tokens,question_lemmas,_,_,_ = stanfordparser(question_text)
                    question_type = get_question_type(question_lemmas)
                    question = {}
                    question['question_id'] = question_id
                    question['question_tokens'] = question_tokens
                    question['question_lemmas'] = question_lemmas
                    question['question_type'] = question_type
#                    question = [question_id,question_tokens,question_lemmas]
                    answer_texts = []
                    answer_idxs = []
                    for answer in qa['answers']:
                        answer_text = answer['text']
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_idxs.append([answer_start,answer_end])
                    answers = {}
                    answers['answer_text'] = answer_texts
                    answers['answer_idxs'] = answer_idxs
#                    answers = [answer_texts,answer_idxs]
                    data.append([doc,question,answers,len(context_tokens)])
    return data
    
def get_2d_spans(text):
    tokens = NLP.tokenizer(text)
    return [(t.idx,t.idx + len(t.text)) for t in tokens]
#def get_2d_spans(text,tokens):
#    cur_idx = 0
#    spans = []
#    for token in tokens:
#        if text.find(token, cur_idx) < 0:
#            print("{} {} {}".format(token, cur_idx, text))
#            raise Exception()
#        else:
#            cur_idx = text.find(token, cur_idx)
#        spans.append((cur_idx, cur_idx + len(token)))
#        cur_idx += len(token)
#    return spans    

def get_word_span(spans, idx):
    start = idx[0]
    stop = idx[1]
    idxs = []
    span_start = -1
    span_stop = -1
    for word_idx, span in enumerate(spans):
        if not (stop <= span[0] or start >= span[1]):
            if start == span[0]:
                span_start = span[0]
            if stop == span[1]:
                span_stop = span[1]
            idxs.append(word_idx)
    assert len(idxs) > 0, "{} {} {}".format(spans, start, stop)
#    if span_start < 0 or span_stop < 0:
#        not_match += 1
#    return (idxs[0],idxs[-1])
#    if span_start < 0 or span_stop < 0:
#        return
    return (idxs[0],idxs[-1])

def get_extrafeatures(s_tokens,s_lemmas,q_tokens,q_lemmas):
    feature_set = []
    q_words_cased = set([w for w in q_tokens])
    q_words_uncased = set([w.lower() for w in q_tokens])
    q_words_lemma = set([w.lower() for w in q_lemmas])
    counter = Counter([w.lower() for w in s_tokens])
    length = len(s_tokens)
    for i in range(len(s_tokens)):
        exact_match = 0
        lower_match = 0
        lemma_match = 0
        if(s_tokens[i] in q_words_cased):
            exact_match = 1.0
        if(s_tokens[i].lower() in q_words_uncased):
            lower_match = 1.0
        if(s_lemmas[i] in q_words_lemma):
            lemma_match = 1.0
        tf = counter[s_tokens[i].lower()] * 1.0/length
        feature_set.append([exact_match,lower_match,lemma_match,tf])
    return feature_set
                   
def vectorize_train(data):
    D = []
    train_ex = []
    for doc,question,answer,doc_len in data:
        spans = doc['context_spans']
        idx = answer['answer_idxs']
        l = get_word_span(spans,idx)
        f = get_extrafeatures(doc['context_tokens'],doc['context_lemmas'],
                              question['question_tokens'],question['question_lemmas'])
#        if(len(a) != 0):
        D.append([doc['context'],doc['context_tokens'],doc['context_poss'],doc['context_ners'],
                  question,answer['answer_text'],answer['answer_tokens'],l,spans,f,doc_len])
    D.sort(key=lambda x:len(x[1]))
    
    for text,tokens,pos,ner,q,a,at,l,spans,f,s_len in D:
        ex = {}
        ex['doc_text'] = text
        ex['doc_tokens'] = tokens
        ex['doc_pos'] = pos
        ex['doc_ner'] = ner
        ex['question'] = q
        ex['answer_text'] = a
        ex['answer_tokens'] = at
        ex['label'] = l
        ex['doc_features'] = f
        ex['doc_spans'] = spans
#        train_ex.append([text,s,pos,ner,[q['question_id'],q['question_tokens']],a,at,l,f,spans])
        train_ex.append(ex)
    return train_ex
    
def vectorize_dev(data):
    D = []
    dev_ex = []
    for doc,question,answer,doc_len in data:
#        a = []
        ll = []
        spans = doc['context_spans']
        f = get_extrafeatures(doc['context_tokens'],doc['context_lemmas'],
                              question['question_tokens'],question['question_lemmas'])
#        for answer in answer[0]:
#            a.append(answer)
        for idx in answer['answer_idxs']:
            l = get_word_span(spans,idx)
            ll.append(l)
        D.append([doc['context'],doc['context_tokens'],doc['context_poss'],doc['context_ners'],
                  question,answer['answer_text'],ll,spans,f,doc_len])
#    D.sort(reverse=True,key=lambda x:x[9])

    for text,tokens,pos,ner,q,aa,ll,spans,f,s_len in D:
        ex = {}
        ex['doc_text'] = text
        ex['doc_tokens'] = tokens
        ex['doc_pos'] = pos
        ex['doc_ner'] = ner
        ex['question'] = q
        ex['answer_text'] = aa
        ex['label'] = ll
        ex['doc_features'] = f
        ex['doc_spans'] = spans
#        dev_ex.append([text,s,pos,ner,[q['question_id'],q['question_tokens']],aa,ll,f,spans])
        dev_ex.append(ex)
    return dev_ex

def text_lower(data):
    return [w.lower() for w in data]