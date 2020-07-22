import numpy as np
import pandas as pd
import nltk
nltk.download('punkt') # one time execution
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import re
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import pickle
import sys

def rank_sentences(text):
  
  sentences = []
  sentences.append(sent_tokenize(text)) 
    
  sentences = [y for x in sentences for y in x] # flatten list
  
  # Extract word vectors
  with open('static/textrank/word_embeddings.p', 'rb') as fp:
    word_embeddings = pickle.load(fp)
  
  # remove punctuations, numbers and special characters
  clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

  # make alphabets lowercase
  clean_sentences = [s.lower() for s in clean_sentences]
  
  # remove stopwords from the sentences
  clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
  
  sentence_vectors = []
  for i in clean_sentences:
    if len(i) != 0:
      v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    else:
      v = np.zeros((100,))
    sentence_vectors.append(v)
    
  # similarity matrix
  sim_mat = np.zeros([len(sentences), len(sentences)])
  
  for i in range(len(sentences)):
    for j in range(len(sentences)):
      if i != j:
        sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
  
  nx_graph = nx.from_numpy_array(sim_mat)
  scores = nx.pagerank(nx_graph)
  
  ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
  
  return ranked_sentences[:10]
  
# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def main():
  return

if __name__ == "__main__":
  main()