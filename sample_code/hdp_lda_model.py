import pandas as pd
from pprint import pprint
import pickle as pkl
import numpy as np

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.corpora import Dictionary
from gensim.models import hdpmodel

# Getting and Cleaning the Data
'''
The documents come the business section of Wired from May 20, 2009 
through August 9, 2017.
'''

def fix_dup(x):
    '''
    Due to changes in the document structure of Wired articles over
    time, the spider picked up older titles twice. This function fixes
    this but eliminated a single title
    E.g., 'This is a sample titleThis is a sample title'
    '''
    if x[:len(x)//2] == x[len(x)//2:]:
        return x[:len(x)//2]
    else:
        return x

# Imports scraped data from JSONs

# Note: One article repeated caused the spider to crash, so the 
# articles after it had to be scraped separately.

full_corpus = pd.read_json('wired/wired.json').append(
    pd.read_json('wired/wired3.json'), ignore_index=True)

# Fixes titles that have duplicates.
full_corpus.title = full_corpus.title.apply(fix_dup)

# Combines all text to be analyzed
full_corpus['doc'] = ' '.join(
    full_corpus.title, full_corpus.body, full_corpus.tags)

# Organize corpus by year in separate objects.
y2017 = full_corpus.doc.loc[full_corpus.date.dt.year == 2017]
y2016 = full_corpus.doc.loc[full_corpus.date.dt.year == 2016]
y2015 = full_corpus.doc.loc[full_corpus.date.dt.year == 2015]
y2014 = full_corpus.doc.loc[full_corpus.date.dt.year == 2014]
y2013 = full_corpus.doc.loc[full_corpus.date.dt.year == 2013]
y2012 = full_corpus.doc.loc[full_corpus.date.dt.year == 2012]
y2011 = full_corpus.doc.loc[full_corpus.date.dt.year == 2011]
y2010 = full_corpus.doc.loc[full_corpus.date.dt.year == 2010]
y2009 = full_corpus.doc.loc[full_corpus.date.dt.year == 2009]
y2008 = full_corpus.doc.loc[full_corpus.date.dt.year == 2008]
y2007 = full_corpus.doc.loc[full_corpus.date.dt.year == 2007]
y2006 = full_corpus.doc.loc[full_corpus.date.dt.year == 2006]
y2005 = full_corpus.doc.loc[full_corpus.date.dt.year == 2005]
y2004 = full_corpus.doc.loc[full_corpus.date.dt.year == 2004]
y2003 = full_corpus.doc.loc[full_corpus.date.dt.year == 2003]
y2002 = full_corpus.doc.loc[full_corpus.date.dt.year == 2002]
y2001 = full_corpus.doc.loc[full_corpus.date.dt.year == 2001]
y2000 = full_corpus.doc.loc[full_corpus.date.dt.year == 2000]
y1999 = full_corpus.doc.loc[full_corpus.date.dt.year == 1999]
y1998 = full_corpus.doc.loc[full_corpus.date.dt.year == 1998]
y1997 = full_corpus.doc.loc[full_corpus.date.dt.year == 1997]
y1996 = full_corpus.doc.loc[full_corpus.date.dt.year == 1996]

# Create an iterable list
years = [y2017,y2016,y2015,y2014,y2013,y2012,y2011,y2010,
         y2009,y2008,y2007,y2006,y2005,y2004,y2003,y2002,
         y2001,y2000,y1999,y1998,y1997,y1996]

def clean(doc):
    '''
    Removes the punctuations, stopwords and normalizes the corpus.
    '''
    
    # Tokenization of words while removing punctuations
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_words = tokenizer.tokenize(doc)
    
    # Finds only nouns
    nouns = ' '.join([token for token, pos in pos_tag(
        tokenized_words) if pos.startswith('N')])
    
    # Removes stop words, including custom stop words
    stop = set(stopwords.words('english')) | set(['company','tech','time'])
    stop_free = ' '.join([i for i in nouns.lower().split() if i not in stop])
    
    # Lemmatization of words
    lemma = WordNetLemmatizer()
    lemmatized_words = ' '.join(
        lemma.lemmatize(word) for word in stop_free.split())

    return lemmatized_words

# Cleans corpus
for i,y in enumerate(years):
    years[i] = [clean(doc) for doc in y]

# Pickle the cleaned data just in case.
with open('years.pkl', 'wb+') as handle:
    pkl.dump(years, handle)

with open('years.pkl', 'rb') as handle:
    years = pkl.load(handle)


# Preparing Document-Term Matrix

# All the text documents combined is known as the corpus. To run any 
# mathematical model on text corpus, it is a good practice to convert 
# it into a matrix representation. The model looks for repeating term
# patterns in the entire document-term matrix.

def find_topics_HDP(corpora):
    '''Finds top 10 topics from corpora, single corpus must be list'''
    topics = []
    
    for corpus in corpora:
        corpus = [doc.split() for doc in corpus]
        
        # Creates the term dictionary of our corpus, 
        # where every unique term is assigned an index.
        dictionary = Dictionary(corpus)
        
        # Converts our corpus into a DTM using the vectorizer 
        # created above.
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in corpus]
        
        # Running and training HDP-LDA model on the document-term matrix.
        
        # The next step is to create an object for the LDA model and 
        # train it on our document-term matrix.

        # The HDP-LDA (or HDP mixture) model is a natural nonparametric
        # generalization of Latent Dirichlet Allocation (LDA), where 
        # the number of topics can be unbounded and learnt from data. 
        # Each group is a document consisting of a bag of words, each 
        # cluster is a topic, and each document is a mixture of topics.

        HDPmodel_ = hdpmodel.HdpModel(doc_term_matrix, id2word = dictionary)
        

        # The top 10 most meaningful topics are chosen from based on 
        # their coherence value.
        topics.append(HDPmodel_.show_topics(num_topics=-1, num_words=10)[:10])
    
    return topics

# Results
topics = find_topics_HDP(years)
pprint(topics)



