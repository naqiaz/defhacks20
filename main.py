import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from urllib.request import urlopen
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import string
from nltk.stem.wordnet import WordNetLemmatizer
import re
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import pdb
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('sentiwordnet')
nltk.download('averaged_perceptron_tagger')

#dataset = pd.read_csv(reviews.tsv,delimite,r='\t',quoting = 3)
base = "https://www.tripadvisor.com/Restaurant_Review-g57821-d492649-Reviews-Pho_75-Herndon_Fairfax_County_Virginia.html"

html = urlopen(base)
soup = BeautifulSoup(html, 'html.parser')
soup = BeautifulSoup(soup.prettify(), 'html.parser')
base = base.replace('.html', '-or{0}.html')

#how to iterate through pages
num_reviews = soup.find('span',class_='reviews_header_count').get_text().strip('')
ind_1 = num_reviews.index('(') + 1
ind_2 = num_reviews.index(')')
num_reviews = int (num_reviews[ind_1:ind_2]) 

all_reviews = []
good_reviews = []

for offset in range(0,num_reviews,10):
  url = base.format(offset)
  html = urlopen(url)
  soup = BeautifulSoup(html, 'html.parser')
  soup = BeautifulSoup(soup.prettify(), 'html.parser')
  for review in soup.find_all('p',class_= 'partial_entry'):
    review = review.get_text().strip('\n')
    if "    More" in review:
      ind_1 = review.index("   More")
      review = review[:ind_1]
    all_reviews.append(review)

pos_reviews = []
def categorize(all_reviews):
    for review in all_reviews:
      lem = WordNetLemmatizer()
      review = review.lower().strip()
      words = nltk.word_tokenize(review)
      stop_words = set(stopwords.words('english'))
      tokens = []
      for w in words:
        if w not in stop_words:
          w = lem.lemmatize(w)
          tokens.append (w)
      tokens = nltk.pos_tag(tokens)
      if (sentiment(tokens)):
        pos_reviews.append(words)

def penn_to_wn(tag):
  if tag.startswith('J'):
    return wn.ADJ
  elif tag.startswith('N'):
    return wn.NOUN
  elif tag.startswith('R'):
    return wn.ADV
  elif tag.startswith('V'):
    return wn.VERB
  return None

def word_sentiment(word,tag):
  wn_tag = penn_to_wn(tag)
  if(wn_tag not in (wn.ADJ,wn.ADV)):
    return 0
  synsets = wn.synsets(word,pos=wn_tag)
  if not synsets:
    return 0
  synset = synsets[0]
  sys = swn.senti_synset(synset.name())
  total_score = sys.pos_score() - sys.neg_score() 
  if (total_score > 0):
    return 1
  if (total_score < 0):
    return -1
  else:
    return 0

def sentiment(tokens):
  score = 0;
  for w in tokens:
    score = score + word_sentiment(w[0],w[1])
  if (score > 0):
      return True
  else:
      return False

def if_food(word):
  syns = wn.synsets(word,pos=wn.NOUN)
  for syn in syns:
    if 'food' in syn.lexname():
      return True
  return False

words_to_remove = ['food', 'delicious', 'mix', 'fare', 'center', 'meal', 'dinner', 'drinks', 'tables', 'taste', 'cup','appetizer','beverage','portion','piece','slice','cuisine','menu','lunch']

def check_review_for_food(pos_reviews):# review should be a list of words
  list_food = []
 
  for review in pos_reviews:
    for w in review:
      if if_food(w): 
        lem = WordNetLemmatizer()
        w = lem.lemmatize(w)
        if w not in words_to_remove:
          list_food.append(w)
  return list_food

    
categorize(all_reviews)
fdist = FreqDist(check_review_for_food(pos_reviews))
print(fdist.most_common(50))

fdist.plot(30,cumulative=False)
plt.show()
pdb.set_trace()
