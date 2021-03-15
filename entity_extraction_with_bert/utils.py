import numpy as np 
import pandas as pd 
import wget
import config


'''
print('Beginning file download with wget module')

url1 = 'https://huggingface.co/bert-base-uncased/resolve/main/config.json'
url2 = 'https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt'
url3 = 'https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin'

wget.download(url1, './weights/bert_base_uncased')
wget.download(url2, './weights/bert_base_uncased')
wget.download(url3, './weights/bert_base_uncased')


print("Done!")
'''

#******************************************************************************



# 47959 sentences containing 35178 different words.
data = pd.read_csv(config.TRAINING_FILE, encoding="latin1")
data = data.fillna(method='ffill')
#data = data[data['Sentence #'] == 'Sentence: 1' ]
print(data.head(25))
#ords = list(set(data["Word"].values))



#******************************************************************************


def process_data(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")

    enc_pos = preprocessing.LabelEncoder()
    enc_tag = preprocessing.LabelEncoder()

    df.loc[:, "POS"] = enc_pos.fit_transform(df["POS"])
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    pos = df.groupby("Sentence #")["POS"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values
    return sentences, pos, tag, enc_pos, enc_tag