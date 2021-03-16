from tqdm import tqdm
import config


def train_fn(model, data_loader, optimizer,device):
    model.train()
    fin_loss = 0
    iou_ = 0
    for X,y in tqdm(data_loader):
        X=X.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        y_ = model(X)
        loss = criterion(y_, y)
        fin_loss += loss.item()
        print(loss)
        loss.backward()
        optimizer.step()
    return fin_loss / len(data_loader)


def eval_fn(model, data_loader,device):
    model.eval()
    fin_loss = 0
    iou_=0
    for X,y in tqdm(data_loader):
        X=X.to(device)
        y=y.to(device) 
        y_=model(X)
        loss=criterion(y_, y)
        fin_loss += loss.item()
    return fin_loss / len(data_loader)

