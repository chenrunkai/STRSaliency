import numpy as np
import torch
from tqdm import tqdm

from models.models import CNN, LSTM
from synth_datasets import MyDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

model_type = "lstm"
dataset_name = "cricket_x"

if model_type =="cnn":
    if dataset_name=="wafer":
        model = CNN(1, 152, dropout=0.25) # 152 for wafer
    elif dataset_name=="ptb":
        model = CNN(1, 187, dropout=0.25) # 187 for ptb
    elif dataset_name=="cricket_x":
        model = CNN(1, 300, dropout=0.25) # 300 for cricket_x
    else:
        model = CNN(1, 100, dropout=0.25) # 100 for synth
elif model_type == "lstm":
    model = LSTM(1, 1, dropout=0, num_layers=3, bidirectional=True)
else:
    raise ValueError("Wrong model type! Should be cnn or lstm. ")
# model = torch.load("./models/trained_models/lstm_remainder_best")
if dataset_name=="wafer":
    dataset = MyDataset("./real_datasets/%s"%dataset_name, 0, int(1524*0.9))
    testset = MyDataset("./real_datasets/%s"%dataset_name, int(1524*0.9), 1524)
elif dataset_name=="ptb":
    dataset = MyDataset("./real_datasets/%s"%dataset_name, 0, int(1456*0.9))
    testset = MyDataset("./real_datasets/%s"%dataset_name, int(1456*0.9), 1456)
elif dataset_name=="cricket_x":
    dataset = MyDataset("./real_datasets/%s"%dataset_name, 0, int(130*0.7))
    testset = MyDataset("./real_datasets/%s"%dataset_name, int(130*0.7), 130)
else:
    dataset = MyDataset("./synth_datasets/%s"%dataset_name, 0, 8000)
    testset = MyDataset("./synth_datasets/%s"%dataset_name, 8000, 10000)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=200)
testloader = torch.utils.data.DataLoader(testset)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)

def train(max_epoch):
    loss_fn = torch.nn.BCELoss().to(device)
    # loss_fn = torch.nn.CrossEntropyLoss().to(device)
    for epoch in range(max_epoch):
        avg_loss = []
        for X, y in dataloader:
            optimizer.zero_grad()
            # print(X.shape)
            output = model(X)
            # print(output)
            # print(y)
            loss = loss_fn(output, y)
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
        if epoch%5==0:
            print("Epoch: %d | loss: %f"%(epoch, np.average(avg_loss)))

def test(loader):
    model.eval()
    results = []
    with torch.no_grad():
        for X, y in loader:
            optimizer.zero_grad()
            output = model(X)
            output = output.item()
            prediction = 1 if output>=0.5 else 0
            label = y.item()
            results.append(1 if prediction==label else 0)
    print("Accuracy: %f"%(np.average(results)))

train(500)
torch.save(model, "./models/trained_models/%s_%s_2"%(model_type, dataset_name))
test(torch.utils.data.DataLoader(dataset))

# Test accuracy
# model = torch.load("./models/trained_models/%s_%s_best"%(model_type, dataset_name))
test(testloader)