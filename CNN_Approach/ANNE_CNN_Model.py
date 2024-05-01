
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from datetime import datetime
import matplotlib.pyplot as plt
# Confusion matrix
from sklearn.metrics import confusion_matrix
import itertools
class TimeSeriesCNN(nn.Module):
    def __init__(self, K):
        super(TimeSeriesCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU()
        )
        
        # Updated the input size for the dense layer based on the conv layer calculations
        self.dense_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128 * 124, 512),  # Updated size
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, K)
        )
    
    def forward(self, X):
        print('forward',flush = True)
        out = self.conv_layers(X)
        out = out.view(out.size(0), -1)  # Flatten the output for the dense layer
        out = self.dense_layers(out)
        print("finished forward",flush = True)
        return out

def stackData(names_data_and_targets):
    res_data = np.empty((0,1000))
    res_target = np.empty((0,1))

    for item in names_data_and_targets:
        data = item['data']
        target = item['target']

        res_data = np.vstack((res_data,data))
        res_target = np.vstack((res_target,target))

    return res_data,res_target

def batch_gd(model,criterion,optimizer,train_loader,test_loader,epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []

        for inputs,targets in train_loader:
            
            inputs = inputs.unsqueeze(1)
            targets = targets.squeeze()
            optimizer.zero_grad()
        

            outputs = model(inputs)
            loss= criterion(outputs,targets)
            
            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

        #get train loss and test loss
        train_loss = np.mean(train_loss)

        model.eval()

        test_loss = []

        for inputs, targets in test_loader:
            inputs = inputs.unsqueeze(1)
            targets = targets.squeeze()

            outputs = model(inputs)

            loss = criterion(outputs,targets)

            test_loss.append(loss.item())

        test_loss = np.mean(test_loss)

        #save losses

        train_losses[it] = train_loss
        test_losses[it] = test_loss

        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
        Test Loss: {test_loss:.4f}, Duration: {dt}',flush = True)

    return train_losses,test_losses, model
def turnToBinary(targets):
    res = np.zeros(shape = (targets.shape))

    for i in range(len(targets)):
        if targets[i][0] > 0:
            res[i][0] = 1

    return res



def plot_confusion_matrix(cm, classes, name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix",flush = True)
    else:
        print('Confusion matrix, without normalization',flush = True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(name)

if __name__ == "__main__":
    '''
    the following is test code'''
    t_start = datetime.now()
    print(f'job started at {t_start}', flush = True)

    #load the pickle dictionary

        # Read and deserialize from file
    print(f"Processing data...",flush = True)
    with open('./data_and_targets.pkl', 'rb') as f:
        arr = pickle.load(f)

    split_index = math.ceil(len(arr) * 0.7)

    training_data = arr[:split_index]
    testing_data = arr[split_index:]

    training_data,training_target = stackData(training_data)
    testing_data,testing_target = stackData(testing_data)

    #turn this classification problem into binary classification
    training_target = turnToBinary(training_target)
    testing_target = turnToBinary(testing_target)

    #loss and optimizer
    model = TimeSeriesCNN(2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    #define batch size
    batch_size = 128

    #initilize training data and target tensors and dataset
    training_data_tensor = torch.tensor(training_data,dtype = torch.float32)
    training_target_tensor = torch.tensor(training_target,dtype=torch.long)
    training_dataset = TensorDataset(training_data_tensor,training_target_tensor)

    #create training dataloader
    train_loader = DataLoader(dataset = training_dataset,batch_size=batch_size, shuffle = True)

    #initialize testing data and target tensors and dataset
    testing_data_tensor = torch.tensor(testing_data,dtype = torch.float32)
    testing_target_tensor = torch.tensor(testing_target,dtype = torch.long)
    testing_dataset = TensorDataset(testing_data_tensor,testing_target_tensor)

    #create testing dataloader
    test_loader = DataLoader(dataset = testing_dataset,batch_size=batch_size,shuffle = True)
    print(f"Finished processing data",flush = True)

    #set epochs 
    epochs = 1000

    print(f"Training model...",flush = True)
    train_losses, test_losses, model = batch_gd(model=model,criterion=criterion,optimizer=optimizer,train_loader=train_loader,test_loader=test_loader,epochs=epochs)
    print(f"Finished training model...",flush = True)
    print(f"Saving model to file...",flush = True)
    torch.save(model.state_dict(),"ANNE_PPG_CNN.pt")
    print(f"Finished saving model to file", flush = True)
    
    #Record accuracy
    model.eval()
    n_correct = 0.
    n_total = 0.
    print(f"Calculating accuracy...",flush = True)
    for inputs,targets in train_loader:
        inputs = inputs.unsqueeze(1)
        targets = targets.squeeze()
        outputs = model(inputs)

        _,predictions = torch.max(outputs,1)

        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    train_acc = n_correct/n_total

    n_correct = 0.
    n_total = 0.

    for inputs,targets in test_loader:
        inputs = inputs.unsqueeze(1)
        targets = targets.squeeze()
        outputs= model(inputs)

        _,predictions= torch.max(outputs,1)

        n_correct += (predictions == targets).sum().item()

        n_total += targets.shape[0]

    test_acc = n_correct/n_total

    print(f"Train acc : {train_acc:.4f}, Test acc : {test_acc:.4f}",flush = True)

    #print confusion matrix
    print(f"Finished calculating accuracy",flush = True)
    x_test = testing_data
    y_test = testing_target
    p_test = np.array([])
    for inputs, targets in test_loader:
        inputs = inputs.unsqueeze(1)
        targets = targets.squeeze()
        # Forward pass
        outputs = model(inputs)

        # Get prediction
        _, predictions = torch.max(outputs, 1)
        
        # update p_test
        p_test = np.concatenate((p_test, predictions.cpu().numpy()))
    cm = confusion_matrix(y_test, p_test)
    print(f"Saving confusion matrix",flush = True)
    plot_confusion_matrix(cm, list(range(2)),'./cm.png')
    print(f"Finished saving confusion matrix", flush = True)
    t_end = datetime.now()
    print(f'job ended at {t_end}',flush = True)

    print(f"Job Duration was {t_end-t_start}",flush = True)

    











