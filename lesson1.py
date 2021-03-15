import pandas as pd 
import numpy as np 


# 47959 sentences containing 35178 different words.
data = pd.read_csv("./data/ner_dataset.csv", encoding="latin1")
data = data.fillna(method='ffill')
#data = data[data['Sentence #'] == 'Sentence: 1' ]

words = list(set(data["Word"].values))

# retrieve sentence 

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
    
    def get_next(self):
        try:
            s = self.data[self.data["Sentence #"] == "Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s["Word"].values.tolist(), s["POS"].values.tolist(), s["Tag"].values.tolist()    
        except:
            self.empty = True
            return None, None, None


sentence_getter = SentenceGetter(data)
sentence_1,_,_ = sentence_getter.get_next()
