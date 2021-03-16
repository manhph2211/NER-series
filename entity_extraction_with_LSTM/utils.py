import pandas as pd 
import numpy as np 
import config
from keras.preprocessing.sequence import pad_sequences
from sklearn import model_selection



# retrieve sentence 
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

    def get_only_se(self):
    	item = self.get_next()
    	words_in_sent=[tup[0] for tup in item]
    	return ' '.join(words_in_sent)


def encode_padding(words,tags,sentences): # 35179 different words, 17 different tags, 47959 sentences
	word2idx = {w: i for i, w in enumerate(words)}
	tag2idx = {t: i for i, t in enumerate(tags)}
	X = [[word2idx[w[0]] for w in s] for s in sentences]
	X = pad_sequences(maxlen=config.MAX_LEN, sequences=X, padding="post", value=len(words)- 1)
	y = [[tag2idx[w[2]] for w in s] for s in sentences]
	y = pad_sequences(maxlen=config.MAX_LEN, sequences=y, padding="post", value=tag2idx["O"])	
	return X,y



def split_data(X,y):
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
	X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
	return X_train,y_train,X_val,y_val,X_test,y_test



def get_data(path="../data/ner_dataset.csv"):

	data = pd.read_csv(path, encoding="latin1")
	data = data.fillna(method='ffill')
	words = list(set(data["Word"].values))
	words.append("ENDPAD")
	tags = list(set(data["Tag"].values))	
	sen_getter = SentenceGetter(data)
	sentences = sen_getter.sentences

	return words, tags, sentences


def to_categorical(y, num_classes): # 1-hot encodes a tensor
    return np.eye(num_classes, dtype='uint8')[y]



if __name__ == '__main__':
	words,tags,sentences = get_data()
	X,y = encode_padding(words,tags,sentences)
	X_train,y_train,X_val,y_val,X_test,y_test = split_data(X,y)
	print(len(X_train))
	print(len(y_train))
