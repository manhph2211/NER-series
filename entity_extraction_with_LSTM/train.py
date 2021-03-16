import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from dataset import my_dataset
from torch.utils.data.dataloader import DataLoader
from model import LSTM
from engine import eval_fn,train_fn
from utils import get_data,split_data,encode_padding
import pandas as pd 
import config


def train():
	words,tags,sentences = get_data()
	X,y = encode_padding(words,tags,sentences)
	X_train,y_train,X_val,y_val,X_test,y_test = split_data(X,y)
	train_dataset = my_dataset(X_train,y_train)
	train_data = DataLoader(train_dataset, batch_size=config.BATCH_SIZE)
	val_dataset = my_dataset(X_val,y_val)
	val_data = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
	test_dataset = my_dataset(X_test,y_test)
	test_data = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = LSTM(input_size=config.MAX_LEN, output_size=config.N_CLASSES, hidden_dim=64, n_layers=2)
	model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
	best_val_loss = 9999
	for epoch in range(config.N_EPOCHS):
	    loss_train_epoch = train_fn(model,train_data,optimizer,device)
	    loss_val_epoch = eval_fn(model,val_data,device)
	    log_epoch = {"epoch": epoch, "train_loss": loss_train_epoch,"val_loss": loss_val_epoch}
	    log.append(log_epoch)
	    df = pd.DataFrame(log)
	    df.to_csv(".weights/logs.csv")
	    #torch.save(model.state_dict(), "./weights/unet" + ".pth")
	    if loss_val_epoch < best_val_loss:
	        best_val_loss = loss_val_epoch
	        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
	    print("Epoch {} || epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f}".format(epoch + 1,loss_train_epoch,loss_val_epoch))


if __name__ == '__main__':
	train()







