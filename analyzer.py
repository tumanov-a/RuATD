#coding=utf-8
import pandas as pd
import numpy as np
import re
import pymorphy2

from tqdm import tqdm
from pymystem3 import Mystem
from tika import parser
from gensim.models import Word2Vec
from string import punctuation
from bs4 import BeautifulSoup
from deeppavlov.models.tokenizers.ru_tokenizer import RussianTokenizer as Tokenizer
from collections import Counter
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from stop_words import get_stop_words

russian_stopwords = get_stop_words('ru')

class Analyzer(object):
    def __init__(self, classifier_path, d2v_model_path, mode):
        self.mystem = Mystem()
        self.morph = pymorphy2.MorphAnalyzer()
        self.punc = punctuation + '«»―'
        self.tokenizer = Tokenizer(stopwords=russian_stopwords, alphas_only=True)
        self.tokens_df = {}
        self.vectors_cache = {}
        self.VECTOR_SIZE = 200
        self.model_dv = Doc2Vec.load(d2v_model_path)
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

    def lemmatize_text(self, text):
        text = re.sub(r'[\)\(»«:.,/;•!Ð£Ñ‡²µÑ¶"=%\^]+', '', text)
        text = re.sub(r'[_—–]+', ' ', text)
        norm_tokens = self.mystem.lemmatize(text)
        norm_tokens = [token.strip('\n') for token in norm_tokens if token not in russian_stopwords and token != " " and token.strip() not in self.punc]
        text = " ".join(norm_tokens)
        return text

    def count_pos(self, text):
        tokens_pos = [self.morph.parse(token)[0].tag.POS for token in self.tokenizer([text])[0] if token.strip() not in self.punc]
        return Counter(tokens_pos)

    def count_letters(self, text, type_):
        if type_ == 'vowel':
            letters = re.findall(r'[АЕЁИОУЫЭЮЯаеёиоуыэюя]{1}', text)
        elif type_ == 'cons':
            letters = re.findall(r'[БВГДЖЗЙКЛМНПРСТФХЦЧШЩбвгджзйклмнпрстфхцчшщ]{1}', text)
        return len(letters)

    def len_text(self, text):
        return len(text)

    def mean_letter_occurance(self, text, type_):
        return np.mean([self.count_letters(token, type_) for token in self.tokenizer([text])[0]])

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

    def count_links(self, text):
        return text.count('http')
    
    def count_space(self, text):
        return len(re.findall(r'\s{1}', text))
    
    def count_lat(self, text):
        return len(re.findall(r'[A-z]{1}', text))
    
    def count_kirr(self, text):
        return len(re.findall(r'[А-я]{1}', text))

    def avg_word_len(self, text):
        return np.mean([len(sent) for sent in self.tokenizer([text])[0]])

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
        df['text'] = df['text'].apply(lambda x: self.remove_invalid_symb(x))
        df['count_links'] = df['text'].apply(lambda x: self.count_links(x))
        df['lemm_text'] = df['text'].apply(lambda text: self.lemmatize_text(text))
        print('Text lemmatized and trash removed...')
       
        df['len_text'] = df['text'].apply(lambda x: self.len_text(x))
        df['count_punct'] = df['text'].apply(lambda x: self.count_punct(x))
        df['count_numbers'] = df['text'].apply(lambda x: self.count_numbers(x))
        df['count_digits'] = df['text'].apply(lambda x: self.count_digits(x))
        df['count_uppercase'] = df['text'].apply(lambda x: self.count_uppercase(x))
        df['count_lowercase'] = df['text'].apply(lambda x: self.count_lowercase(x))
        df['avg_word_len'] = df['text'].apply(lambda x: self.avg_word_len(x))
        df['mean_vowel_occurance'] = df['text'].apply(lambda text: self.mean_letter_occurance(text, 'vowel'))
        df['mean_consonant_occurance'] = df['text'].apply(lambda text: self.mean_letter_occurance(text, 'cons'))
        df['count_space'] = df['text'].apply(lambda x: self.count_space(x))
        df['count_space_to_len'] = df['count_space'] / df['len_text']
                                             
        df['count_kirr'] = df['text'].apply(lambda x: self.count_kirr(x))
        df['count_lat'] = df['text'].apply(lambda x: self.count_lat(x))
                                             
        df['lat_symb_density'] = df['count_lat'] / (df['count_kirr'] + df['count_lat'])
                                             
        print('Features is calculated...')
        
        df['count_POS'] = df['text'].apply(lambda x: self.count_pos(x))
        
        df.reset_index(inplace=True, drop=True)
        
        for ind in df.index:
            count_pos = df.loc[ind, 'count_POS']
            for pos, count_ in count_pos.items():
                col_name = 'POS_' + str(pos)
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
        tokenized_corpus = self.tokenizer(np.array(df['lemm_text']))
        return tokenized_corpus

    def __train_wv_model(self, tokenized_corpus, VECTOR_SIZE):
        self.model = Word2Vec(sentences=tokenized_corpus, vector_size=VECTOR_SIZE, workers=10, alpha=0.025, epochs=10, min_count=2, max_vocab_size=100000)
        self.model.save("word2vec.model")

    def __sentence_vectors(self, tokenized_corpus, VECTOR_SIZE):
        sentence_vectors = np.empty((len(tokenized_corpus), VECTOR_SIZE))
        a, n = 0, 0

        for i, tokenized_sentence in tqdm(enumerate(tokenized_corpus)):
            sentence_vector = np.empty((len(tokenized_corpus), VECTOR_SIZE))

            for j, token in enumerate(tokenized_sentence):
                try:
                    if token in self.vectors_cache:
                        token_vector = self.vectors_cache[token]
                    else:
                        token_vector = self.model.wv[token]
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

    def __add_wv_vectors(self, df):
        tokenized_corpus = self.tokenize_corpus(df)
        print('Train word2vec is complete...')
        wv_vectors = self.sentence_vectors(tokenized_corpus, self.VECTOR_SIZE)
        wv_df = pd.DataFrame(wv_vectors, columns=['wv_' + str(i) for i in range(wv_vectors.shape[1])])
        df.reset_index(inplace=True, drop=True)
        df = pd.concat([df, wv_df], axis=1)
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

        d2v_vectors = self.document_vectors(len(tagged_tokenized_corpus), self.VECTOR_SIZE)

        dv_df = pd.DataFrame(d2v_vectors, columns=['dv_' + str(i) for i in range(d2v_vectors.shape[1])])
        df.reset_index(inplace=True, drop=True)
        df = pd.concat([df, dv_df], axis=1)
        return df

    def postprocess(self, df):
        df['lemm_text'] = df['lemm_text'].apply(lambda x: re.sub(r'[\)\(»«:.,/;•!Ð£Ñ‡²µÑ¶]{1}"=%', '', x))
        df['lemm_text'] = df['lemm_text'].apply(lambda x: re.sub(r'[_—–]{1}', ' ', x))
        df['lemm_text'] = df['lemm_text'].apply(lambda x: re.sub(r'\s+', ' ', x))
        df.fillna(0, inplace=True)
        return df

    def predict_doc_label(self, doc_paths):  
        data = self.preprocess_pipeline(data)
        data = self.add_d2v_vectors(data)
        data = self.postprocess(data)
        return data
            
global cols
cols = ['count_links',
         'lemm_text',
         'len_text',
         'count_punct',
         'count_numbers',
         'count_digits',
         'count_uppercase',
         'count_lowercase',
         'avg_word_len',
         'mean_vowel_occurance',
         'mean_consonant_occurance',
         'count_space',
         'count_space_to_len',
         'count_kirr',
         'count_lat',
         'lat_symb_density',
         'POS_NOUN',
         'POS_ADJF',
         'POS_None',
         'POS_VERB',
         'POS_INFN',
         'POS_ADVB',
         'POS_GRND',
         'POS_PREP',
         'POS_PRTF',
         'POS_PRTS',
         'POS_ADJS',
         'POS_INTJ',
         'POS_COMP',
         'POS_PRCL',
         'POS_PRED',
         'POS_CONJ',
         'POS_NUMR',
         'POS_NPRO',
         'dv_0',
         'dv_1',
         'dv_2',
         'dv_3',
         'dv_4',
         'dv_5',
         'dv_6',
         'dv_7',
         'dv_8',
         'dv_9',
         'dv_10',
         'dv_11',
         'dv_12',
         'dv_13',
         'dv_14',
         'dv_15',
         'dv_16',
         'dv_17',
         'dv_18',
         'dv_19',
         'dv_20',
         'dv_21',
         'dv_22',
         'dv_23',
         'dv_24',
         'dv_25',
         'dv_26',
         'dv_27',
         'dv_28',
         'dv_29',
         'dv_30',
         'dv_31',
         'dv_32',
         'dv_33',
         'dv_34',
         'dv_35',
         'dv_36',
         'dv_37',
         'dv_38',
         'dv_39',
         'dv_40',
         'dv_41',
         'dv_42',
         'dv_43',
         'dv_44',
         'dv_45',
         'dv_46',
         'dv_47',
         'dv_48',
         'dv_49',
         'dv_50',
         'dv_51',
         'dv_52',
         'dv_53',
         'dv_54',
         'dv_55',
         'dv_56',
         'dv_57',
         'dv_58',
         'dv_59',
         'dv_60',
         'dv_61',
         'dv_62',
         'dv_63',
         'dv_64',
         'dv_65',
         'dv_66',
         'dv_67',
         'dv_68',
         'dv_69',
         'dv_70',
         'dv_71',
         'dv_72',
         'dv_73',
         'dv_74',
         'dv_75',
         'dv_76',
         'dv_77',
         'dv_78',
         'dv_79',
         'dv_80',
         'dv_81',
         'dv_82',
         'dv_83',
         'dv_84',
         'dv_85',
         'dv_86',
         'dv_87',
         'dv_88',
         'dv_89',
         'dv_90',
         'dv_91',
         'dv_92',
         'dv_93',
         'dv_94',
         'dv_95',
         'dv_96',
         'dv_97',
         'dv_98',
         'dv_99',
         'dv_100',
         'dv_101',
         'dv_102',
         'dv_103',
         'dv_104',
         'dv_105',
         'dv_106',
         'dv_107',
         'dv_108',
         'dv_109',
         'dv_110',
         'dv_111',
         'dv_112',
         'dv_113',
         'dv_114',
         'dv_115',
         'dv_116',
         'dv_117',
         'dv_118',
         'dv_119',
         'dv_120',
         'dv_121',
         'dv_122',
         'dv_123',
         'dv_124',
         'dv_125',
         'dv_126',
         'dv_127',
         'dv_128',
         'dv_129',
         'dv_130',
         'dv_131',
         'dv_132',
         'dv_133',
         'dv_134',
         'dv_135',
         'dv_136',
         'dv_137',
         'dv_138',
         'dv_139',
         'dv_140',
         'dv_141',
         'dv_142',
         'dv_143',
         'dv_144',
         'dv_145',
         'dv_146',
         'dv_147',
         'dv_148',
         'dv_149',
         'dv_150',
         'dv_151',
         'dv_152',
         'dv_153',
         'dv_154',
         'dv_155',
         'dv_156',
         'dv_157',
         'dv_158',
         'dv_159',
         'dv_160',
         'dv_161',
         'dv_162',
         'dv_163',
         'dv_164',
         'dv_165',
         'dv_166',
         'dv_167',
         'dv_168',
         'dv_169',
         'dv_170',
         'dv_171',
         'dv_172',
         'dv_173',
         'dv_174',
         'dv_175',
         'dv_176',
         'dv_177',
         'dv_178',
         'dv_179',
         'dv_180',
         'dv_181',
         'dv_182',
         'dv_183',
         'dv_184',
         'dv_185',
         'dv_186',
         'dv_187',
         'dv_188',
         'dv_189',
         'dv_190',
         'dv_191',
         'dv_192',
         'dv_193',
         'dv_194',
         'dv_195',
         'dv_196',
         'dv_197',
         'dv_198',
         'dv_199']