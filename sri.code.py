import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
if torch.cuda.is_available():
 device = torch.device("cuda")
 print( torch.cuda.device_count())
 print('Available:', torch.cuda.get_device_name(0))
else:
 print('No GPU available, using the CPU instead.')
 device = torch.device("cpu"
!pip install wget
!pip install transformers
url_train='https://groups.csail.mit.edu/sls/downloads/movie/engtrain.bio'
url_test='https://groups.csail.mit.edu/sls/downloads/movie/engtest.bio'
import wget
import os
wget.download(url_train)
wget.download(url_test)
import csv
sentences = []
labels = []
tokens = []
token_labels = []
unique_labels = set()
with open("./engtrain.bio", newline = '') as lines:
 line_reader = csv.reader(lines, delimiter='\t')
 for line in line_reader:
 if line == []:
 sentences.append(tokens)
 labels.append(token_labels)
 tokens = []
 token_labels = []
 else:
 tokens.append(line[1])
 token_labels.append(line[0])
 unique_labels.add(line[0])
[ print(' '.join(sentences[i])) for i in range(10)]
' '.join(sentences[1])
pd.DataFrame({"Word":sentences[1],"Labels":labels[1]})
from transformers import BertTokenizer
import numpy as np
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.figure(figsize=(24,24))
plt.rcParams["figure.figsize"] = (10,5)
# Plot the distribution of comment lengths.
sns.distplot(TokenLength, kde=False, rug=False,color='plum')
plt.title('Sentence Lengths')
plt.xlabel('Sentence Length')
plt.ylabel('# of Sentences');
import spacy
nlp = spacy.load('en_core_web_sm')
text = u'I will visit Paris on November 2021'
doc = nlp(text)
def displayEntities(doc):
 if doc.ents:
 for entity in doc.ents:
 print('Entity: {}, Label: {}, Explanation: {}'.format(entity.text, entity.label_, 
spacy.explain(entity.label_)))
 else:
 print('[INFO] No Entity found!')
displayEntities(doc)
from spacy.tokens import Span
newText = u'SpaceX is going to lead NASA soon!'
newDoc = nlp(newText)
ORG = newDoc.vocab.strings[u'ORG']
newEntity = Span(newDoc, 0, 1, label=ORG)
newDoc.ents = list(newDoc.ents)+[newEntity]
displayEntities(newDoc)
