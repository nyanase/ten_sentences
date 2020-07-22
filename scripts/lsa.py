from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
from newspaper import Article

def main(url):
  article_content = get_content_from_url(url)
  ranked_sentences = rank_lsa(article_content)
  return ranked_sentences

def rank_lsa(article, num_sentences=5):  
  doc = []
  doc.append(sent_tokenize(article))
  doc = [y for x in doc for y in x] # flatten list
      
  # Converting each document into an vector
  vectorizer = CountVectorizer()
  bag_of_words = vectorizer.fit_transform(doc)
  num_sentences = min(num_sentences, bag_of_words.shape[1])

  #Singular value decomposition
  #This process encodes our original data into topic encoded data
  svd = TruncatedSVD(n_components = bag_of_words.shape[1]-1)
  lsa = svd.fit_transform(bag_of_words)

  df = pd.DataFrame(lsa)
  df.insert(0, "sentence", doc)

  # Get the sentences in order of scores
  ranked_sentences = []
  
  num_topics = min(len(doc), num_sentences)

  for i in range(num_topics):
      ranked = df[i].nlargest(num_topics).reset_index()[i]
      cur_index = 0
      while True:    
          sent = df.loc[df[i] == ranked[cur_index]]["sentence"].values[0]
          if sent not in ranked_sentences:
              ranked_sentences.append(sent)
              break
          cur_index += 1

  return ranked_sentences

def get_content_from_url(url):
  try:
    article = Article(url)
    article.download()
    article.parse()
  except:
    return False
  return article.text
  

if __name__ == "__main__":
  main()
