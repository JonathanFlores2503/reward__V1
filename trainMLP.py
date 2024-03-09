# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import matplotlib.pyplot as plt
import mlpArchitecture 


"""MLP (pp. 5)"""

csv_features_path = "TraningData/features.csv"
cvs_labels_path = "TraningData/labales.csv"
nameModel = "MLP_I1_v1"

I_1_features = np.genfromtxt(csv_features_path, delimiter=',')
I_1_features = torch.from_numpy(I_1_features).to('cuda') 
I_1_features = I_1_features.to(dtype=torch.float32)

I_1_labels = np.genfromtxt(cvs_labels_path, delimiter=',')
I_1_labels = torch.from_numpy(I_1_labels).to('cuda') 
I_1_features = I_1_features.to(dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

I_1_features = I_1_features.to(device)
I_1_labels = I_1_labels.to(device)

dataset = TensorDataset(I_1_features, I_1_labels)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = mlpArchitecture.shallowNN().to(device)

# Adam (5e−5 learning rate with 0.001 weight decay) optimizer configuration (pp. 5)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.001)

# BCE loss (pp. 5)
criterion = nn.BCELoss()

num_epochs = 1000
losses = []
accuracies = []

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).view(-1, 1).float()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward y optimiza
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = outputs.round()  
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct / total
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}, Accuracy: {epoch_acc}')
    
    losses.append(epoch_loss)
    accuracies.append(epoch_acc)
    
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Loss')
plt.plot(accuracies, label='Accuracy')
plt.title('Trains Accuracy/Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()

plt.savefig(( "Graphs/" + nameModel + ".jpg"), bbox_inches='tight')
plt.show()

torch.save(model.state_dict(), ("Models/" + nameModel + ".pth"))