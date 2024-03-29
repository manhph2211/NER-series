import torch
import torchvision
from utils import get_data,split_data,encode_padding, to_categorical
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import config


class my_dataset(Dataset):

	def __init__(self,X,y):
		self.X=X
		self.y=to_categorical(y,config.N_CLASSES)


	def __getitem__(self,idx):
	
		X_idx=self.X[idx]
		# X_idx=[[x] for x in X_idx]
		X_idx=torch.LongTensor(X_idx)
		y_idx=self.y[idx]
		y_idx=torch.LongTensor(y_idx)
		return X_idx,y_idx

	def __len__(self):
		return len(self.X)




if __name__ == '__main__':

	words,tags,sentences = get_data()
	X,y = encode_padding(words,tags,sentences)
	X_train,y_train,X_val,y_val,X_test,y_test = split_data(X,y)
	train_dataset = my_dataset(X_train,y_train)
	train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE)
	print(len(train_dataloader))
	item = iter(train_dataloader).next()
	print(item[1].shape)

