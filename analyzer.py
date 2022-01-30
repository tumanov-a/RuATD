#coding=utf-8
import pandas as pd
import numpy as np
import re
import pymorphy2
import gensim

from tqdm import tqdm
from pymystem3 import Mystem
from gensim.models import Word2Vec
from string import punctuation
from collections import Counter
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from stop_words import get_stop_words
from razdel import tokenize as razdel_tokenize


russian_stopwords = get_stop_words('ru')

class Analyzer(object):
    def __init__(self, mode):
        self.mystem = Mystem()
        self.morph = pymorphy2.MorphAnalyzer()
        self.punc = punctuation + '«»―'
        self.tokens_df = {}
        self.vectors_cache = {}
        self.VECTOR_SIZE = 100
        self.mode = mode
        self.pos_cols = ['POS_NOUN',
                         'POS_None',
                         'POS_ADJF',
                         'POS_VERB',
                         'POS_PRTF',
                         'POS_ADVB',
                         'POS_INFN',
                         'POS_CONJ',
                         'POS_PRTS',
                         'POS_GRND',
                         'POS_ADJS',
                         'POS_PRED',
                         'POS_PREP',
                         'POS_NUMR',
                         'POS_PRCL',
                         'POS_INTJ',
                         'POS_NPRO',
                         'POS_COMP']

    def tokenize_text(self, text):
        text = re.sub(r'[\)\(»«:.,/;•!Ð£Ñ‡²µÑ¶"=%\^…]+', '', text)
        text = re.sub(r'[_—–]+', ' ', text)
        tokenized_text = [word.text.strip(punctuation) for word in razdel_tokenize(text)]
        tokenized_text = [word.lower() for word in tokenized_text if word and word not in self.punc]
        return ' '.join(tokenized_text)

    def lemmatize_text(self, text):
        tokenized_text = self.tokenize_text(text).split()
        normalized_text = [self.morph.parse(word)[0].normal_form for word in tokenized_text]
        return ' '.join(normalized_text)

    def return_pos(self, text):
        tokens_pos = [self.morph.parse(token.text)[0].tag.POS for token in razdel_tokenize(text) if token.text.strip() not in self.punc]
        return tokens_pos

    def return_sent_from_pos(self, tokens_pos):
        if tokens_pos:
            return ' '.join(map(str, tokens_pos))
        else:
            return ''

    def count_pos(self, tokens_pos):
        return Counter(tokens_pos)

    def count_letters(self, text, type_, lang_):
        if type_ == 'vowel' and lang_ == 'ru':
            letters = re.findall(r'[АЕЁИОУЫЭЮЯаеёиоуыэюя]{1}', text)
        elif type_ == 'cons' and lang_ == 'ru':
            letters = re.findall(r'[БВГДЖЗЙКЛМНПРСТФХЦЧШЩбвгджзйклмнпрстфхцчшщ]{1}', text)
        elif type_ == 'vowel' and lang_ == 'en':
            letters = re.findall(r'[AEIOUYaeiouy]{1}', text)
        elif type_ == 'cons' and lang_ == 'en':
            letters = re.findall(r'[BCDFGHJKLMNPQRSTVWXZbcdfghjklmnpqrstvwxz]{1}', text)
        return len(letters)

    def len_text(self, text):
        return len(text)

    def mean_letter_occurance(self, text, type_, lang_):
        return np.mean([self.count_letters(word.text, type_, lang_) for word in razdel_tokenize(text)])

    def count_punct(self, text):
        return len(re.findall('[‟„”“№!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~«»―]+', text))

    def count_numbers(self, text):
        return len(re.findall(r'[0-9]+', text))

    def count_digits(self, text):
        return len(re.findall(r'[0-9]{1}', text))

    def count_uppercase(self, text):
        return len(re.findall(r'[A-ZА-Я]{1}', text))

    def count_lowercase(self, text):
        return len(re.findall(r'[a-zа-я]{1}', text))
    
    def count_space(self, text):
        return len(re.findall(r'\s{1}', text))
    
    def count_lat(self, text):
        return len(re.findall(r'[A-z]{1}', text))
    
    def count_kirr(self, text):
        return len(re.findall(r'[А-я]{1}', text))

    def avg_word_len(self, text):
        return np.mean([len(word.text) for word in razdel_tokenize(text)])

    def remove_invalid_symb(self, text):
        map_ = {
                ord('\uf0b4'): '',
                ord('\uf02b'): '',
                ord('\uf03d'): '',
                ord('\uf0d7'): '',
                ord('\uf02d'): '',
                ord('\uf0bb'): '',
                ord('\xad'):   '',
                ord('\uf036'): '',
                ord('\uf041'): '',
                ord('\uf04b'): '',
                ord('\uf04f'): '',
                ord('\uf049'): '',
                ord('\uf046'): '',
                ord('\uf02a'): '',
                ord('\uf068'): '',
                ord('\uf062'): '',
                ord('\uf067'): '',
                ord('\uf061'): '',
                ord('\uf06a'): '',
                ord('\uf02b'): '',
                ord('\u00ad'): '',
                ord('\uf053'): '',
               }

        text = text.translate(map_)
        return text
    
    def preprocess_pipeline(self, df):
        df[self.analyzed_col].fillna('', inplace=True)
        df[self.analyzed_col] = df[self.analyzed_col].apply(lambda x: self.remove_invalid_symb(str(x)))
        df['lemm_' + self.analyzed_col] = df[self.analyzed_col].apply(lambda text: self.lemmatize_text(text))
        print('Text lemmatized and trash removed...')
       
        df['len_text_' + self.analyzed_col] = df[self.analyzed_col].apply(lambda x: self.len_text(x))
        df['count_punct_' + self.analyzed_col] = df[self.analyzed_col].apply(lambda x: self.count_punct(x)) / df['len_text_' + self.analyzed_col]
        df['count_numbers_' + self.analyzed_col] = df[self.analyzed_col].apply(lambda x: self.count_numbers(x)) / df['len_text_' + self.analyzed_col]
        df['count_digits_' + self.analyzed_col] = df[self.analyzed_col].apply(lambda x: self.count_digits(x)) / df['len_text_' + self.analyzed_col]
        df['count_uppercase_' + self.analyzed_col] = df[self.analyzed_col].apply(lambda x: self.count_uppercase(x)) / df['len_text_' + self.analyzed_col]
        df['count_lowercase_' + self.analyzed_col] = df[self.analyzed_col].apply(lambda x: self.count_lowercase(x)) / df['len_text_' + self.analyzed_col]
        df['avg_word_len_' + self.analyzed_col] = df[self.analyzed_col].apply(lambda x: self.avg_word_len(x))
        df['mean_ru_vowel_occurance_' + self.analyzed_col] = df[self.analyzed_col].apply(lambda text: self.mean_letter_occurance(text, 'vowel', 'ru'))
        df['mean_ru_consonant_occurance_' + self.analyzed_col] = df[self.analyzed_col].apply(lambda text: self.mean_letter_occurance(text, 'cons', 'ru'))
        df['mean_en_vowel_occurance_' + self.analyzed_col] = df[self.analyzed_col].apply(lambda text: self.mean_letter_occurance(text, 'vowel', 'en'))
        df['mean_en_consonant_occurance_' + self.analyzed_col] = df[self.analyzed_col].apply(lambda text: self.mean_letter_occurance(text, 'cons', 'en'))
        df['count_space_' + self.analyzed_col] = df[self.analyzed_col].apply(lambda x: self.count_space(x)) / df['len_text_' + self.analyzed_col]
                                             
        df['count_kirr_' + self.analyzed_col] = df[self.analyzed_col].apply(lambda x: self.count_kirr(x)) / df['len_text_' + self.analyzed_col]
        df['count_lat_' + self.analyzed_col] = df[self.analyzed_col].apply(lambda x: self.count_lat(x)) / df['len_text_' + self.analyzed_col]        
        
        print('Features is calculated...')

        df['POS_' + self.analyzed_col] = df[self.analyzed_col].apply(lambda x: self.return_pos(x))
        df['count_POS_' + self.analyzed_col] = df['POS_' + self.analyzed_col].apply(lambda x: self.count_pos(x))
        df['sent_POS_' + self.analyzed_col] = df['POS_' + self.analyzed_col].apply(lambda x: self.return_sent_from_pos(x))
        
        df.reset_index(inplace=True, drop=True)
        
        for ind in df.index:
            count_pos = df.loc[ind, 'count_POS_' + self.analyzed_col]
            for pos, count_ in count_pos.items():
                col_name = 'POS_' + str(pos) + '_' + self.analyzed_col
                df.loc[ind, col_name] = count_
                
        if self.mode == 'inference':
            for col in self.pos_cols:
                if col not in list(df):
                    df[col] = np.zeros(df.shape[0])
                
        print('Count POS is complete...')
        return df

    def calc_tf(self, tokens, token):
        tf =  tokens.count(token) / len(tokens)
        return tf

    def calc_tokens_df(self, corpus):
        for sent in corpus:
            for token in set(sent):
                if token in self.tokens_df:
                    self.tokens_df[token] += 1
                else:
                    self.tokens_df[token] = 1
        return self.tokens_df

    def calc_df(self, token):
        return self.tokens_df[token]

    def tokenize_corpus(self, df):
        tokenized_corpus = [[token.text.strip(punctuation) for token in razdel_tokenize(row)] for row in df[self.analyzed_col]]
        return tokenized_corpus

    def train_wv_model(self, tokenized_corpus, VECTOR_SIZE):
        self.model = Word2Vec(sentences=tokenized_corpus, vector_size=VECTOR_SIZE, workers=10, alpha=0.025, epochs=10, min_count=2, max_vocab_size=100000)
        self.model.save("word2vec.model")
    
    def train_fast_text_model(self, tokenized_corpus, VECTOR_SIZE):
        self.ft_model = gensim.models.FastText(tokenized_corpus, vector_size=VECTOR_SIZE, workers=10, epochs=10, min_count=2, max_vocab_size=100000)
        self.ft_model.save('fastttext.model')

    def sentence_vectors(self, tokenized_corpus, VECTOR_SIZE):
        sentence_vectors = np.empty((len(tokenized_corpus), VECTOR_SIZE))
        a, n = 0, 0

        for i, tokenized_sentence in tqdm(enumerate(tokenized_corpus)):
            sentence_vector = np.empty((len(tokenized_corpus), VECTOR_SIZE))

            for j, token in enumerate(tokenized_sentence):
                try:
                    if token in self.vectors_cache:
                        token_vector = self.vectors_cache[token]
                    else:
                        token_vector = self.ft_model.wv[token]
                        self.vectors_cache[token] = token_vector

                    tf = self.calc_tf(tokenized_sentence, token)
                    idf = np.log(len(tokenized_corpus) / self.calc_df(token))
                    tfidf = tf * idf
                    sentence_vector[j] = token_vector * tfidf
                    a += 1
                except:
                    n += 1
                    pass
            pooled_sentence_vector = sentence_vector.mean(axis=0)
            sentence_vectors[i] = pooled_sentence_vector

        print(f'Ok tokens: {a} | Not ok tokens: {n}')

        return sentence_vectors

    def add_ft_vectors(self, df):
        tokenized_corpus = self.tokenize_corpus(df)
        self.calc_tokens_df(tokenized_corpus)
        if self.mode == 'train':
            self.train_fast_text_model(tokenized_corpus, self.VECTOR_SIZE)
        else:
            self.ft_model = gensim.models.FastText.load('fastttext.model')
        print('Train fasttext model is complete...')
        ft_vectors = self.sentence_vectors(tokenized_corpus, self.VECTOR_SIZE)
        ft_df = pd.DataFrame(ft_vectors, columns=['ft_' + str(i) for i in range(ft_vectors.shape[1])])
        df.reset_index(inplace=True, drop=True)
        df = pd.concat([df, ft_df], axis=1)
        return df

    def document_vectors(self, n_docs, VECTOR_SIZE):
        doc_vectors = np.empty((n_docs, VECTOR_SIZE))
        v_i = 0

        for vector in self.model_dv.dv:
            doc_vectors[v_i] = vector
            v_i += 1
            if v_i == n_docs:
                break
        return doc_vectors

    def train_d2v_model(self, documents, VECTOR_SIZE):
        self.model_dv = Doc2Vec(documents, vector_size=VECTOR_SIZE, min_count=2, workers=10, epochs=10)
        self.model_dv.save("doc2vec.model")

    def add_d2v_vectors(self, df):
        tokenized_corpus = self.tokenize_corpus(df)
        tagged_tokenized_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_corpus)]
        if self.mode == 'train':
            self.train_d2v_model(tagged_tokenized_corpus, self.VECTOR_SIZE)
        else:
            self.model_dv = Doc2Vec.load("doc2vec.model")

        d2v_vectors = self.document_vectors(len(tagged_tokenized_corpus), self.VECTOR_SIZE)

        dv_df = pd.DataFrame(d2v_vectors, columns=[self.analyzed_col + '_dv_' + str(i) for i in range(d2v_vectors.shape[1])])
        df.reset_index(inplace=True, drop=True)
        df = pd.concat([df, dv_df], axis=1)
        return df

    def postprocess(self, df):
        df['lemm_' + self.analyzed_col].fillna('', inplace=True)
        #df['lemm_' + self.analyzed_col] = df['lemm_' + self.analyzed_col].apply(lambda x: re.sub(r'[\)\(»«:.,/;•!Ð£Ñ‡²µÑ¶"=%]{1}', '', x))
        df['lemm_' + self.analyzed_col] = df['lemm_' + self.analyzed_col].apply(lambda x: re.sub(r'[_—–]{1}', ' ', x))
        df['lemm_' + self.analyzed_col] = df['lemm_' + self.analyzed_col].apply(lambda x: re.sub(r'\s+', ' ', x))
        return df

    def data_processing(self, data, analyzed_col):
        self.analyzed_col = analyzed_col
        data = self.preprocess_pipeline(data)
        data = self.add_d2v_vectors(data)
        data = self.postprocess(data)
        return data
    