import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv

#%% 
# Page 4 Set generation F_n, F_a
featuresI3DDir_path = "E:/UCF-Crime/I3D_Features/framesInputFullFps"
nameModel = "REWARD15-10"
featuresI3D_classes = os.listdir(featuresI3DDir_path)
F_n = []
F_a = []
keyWord = 'feature'

for classesI3D in featuresI3D_classes:
    featuresOneClasse_path = os.path.join(featuresI3DDir_path, classesI3D)
    featuresOneClasseFiles = os.listdir(featuresOneClasse_path)
    
    for featureName in featuresOneClasseFiles:
        feature_path = os.path.join(featuresOneClasse_path, featureName)
        # print(feature_path)  
        
        features = np.load(feature_path)
        if classesI3D == "Normal":
            F_n.append(features[keyWord])
        else:
            F_a.append(features[keyWord])

F_n = np.concatenate(F_n, axis=1)

print("F_n dimension: ", F_n.shape) # F_n = (1, N = [N_n*T], 1024)

F_n_list = F_n[0].tolist()
with open('F_n_Features.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(F_n_list)
# 0/0
dimensions = list(map(lambda x: x.shape, F_a)) # F_a = {F^t_aj}_j=1^N_a,t=1^T
print("F_a contains ", len(F_a), "\n"
      "with the following dimensions:", dimensions) # F_a = N_a[j:N_a]
                                                           # |-->(1,[t:T],1024)

# Page 4 Distance Calculation: Distance layer

F_n = torch.from_numpy(F_n).to('cuda').squeeze(0) # [N x P]
F_a = [torch.from_numpy(arr).to('cuda').squeeze(0) for arr in F_a] # ([t:T],1024)
deltaDistances = []

for N_a in range(len(F_a)):
    f_a = F_a[N_a] #[T x P]
    diff_squared = (f_a[:, None, :] - F_n[None, :, :]).pow(2) # Calculates squared difference between each pair of points.
    distanceLayer = diff_squared.sum(dim=2).sqrt() # Sums squared differences and calculates square root to get Euclidean 
                                                   # distance. 
                                                   # Delta = [delta_ti]_{t=1}^{T, N} {i=1} = N_a * [T x N] 
    deltaDistances.append(distanceLayer)
#%%    
# Page 4 Distance Calculation: Sorting layer, average pooling layer
k = 20
thr_lambda = 0.8
I_1_features = []
I_1_labels = []

for videoAnormal in range(len(F_a)):

    sorted_delta, _ = deltaDistances[videoAnormal].sort(dim=1, descending=False) # Sorter layer [T x N]
    averageSortedDelta = sorted_delta[:, :k].mean(dim=1) # Average pooling layer KNN, [T]
    
    # Normalization of distance values around zero
    mean_delta = averageSortedDelta.mean() # (1/T) x sum(delta_t, t=1 to T)
    delta_tilde = averageSortedDelta - mean_delta # delta_tilde_t = *delta_t - mean_delta, [T]; ecuation (1)
    
    # delta_tilde = delta_tilde.cpu().numpy()
    
    # Acumulative sum operation and a ReLU activation function
    D_t = torch.zeros_like(delta_tilde)
    for i in range(1, len(delta_tilde)):
        D_t[i] = max(0, D_t[i-1] + delta_tilde[i]) # Acumulative sum and ReLU activation function
                                                   # Ecuation (2),  [T]
    # Normalization of D_t by its largest value,
    testerDt = D_t / D_t.max() # Ecuation (3),  [T[0:1]] 
    
    # Selection of segments greater than lambda
    D_lambda_thresholded = torch.where(testerDt < thr_lambda, torch.zeros_like(testerDt), testerDt)
    
    
    # MLP and Final Selection:
    for index in  range(D_lambda_thresholded.numel()):
        I_1_features.append(F_a[videoAnormal][index])
        if D_lambda_thresholded[index] != 0:
            I_1_labels.append(1)
            continue
        I_1_labels.append(0)

# MLP and Final Selection:
for index in  range(len(F_n)):
    I_1_features.append(F_n[index])
    I_1_labels.append(0)
assert(len(I_1_features) == len(I_1_labels))
# 0/0
#%% MLP (pp. 5)

I_1_features = torch.stack(I_1_features)
I_1_labels = torch.tensor(I_1_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

I_1_features = I_1_features.to(device)
I_1_labels = I_1_labels.to(device)

dataset = TensorDataset(I_1_features, I_1_labels)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class shallowNN(nn.Module):
    def __init__(self):
        super(shallowNN, self).__init__()

        self.fc1 = nn.Linear(1024, 1000)  # First hidden layer with 1000 neurons (pp. 5)
        self.fc2 = nn.Linear(1000, 1000)  # Second hidden layer with 1000 neurons (pp. 5)
        self.fc3 = nn.Linear(1000, 1)     # Output layer with binary classification (pp.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = torch.sigmoid(self.fc3(x))  
        return x

model = shallowNN().to(device)

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

plt.savefig((nameModel + ".jpg"), bbox_inches='tight')
plt.show()

torch.save(model.state_dict(), (nameModel + ".pth"))
#%% Final smoothing and selection procedure (pp. 5)
model = shallowNN().to(device)
model.load_state_dict(torch.load((nameModel + ".pth"), map_location=device))
model.eval()

F_a_pt = []
for videosAnormals in range(len(F_a)): 
    videoAnormal_pt = F_a[videosAnormals]
    with torch.no_grad():
        snnipetAnormal_pt = model(videoAnormal_pt)
    F_a_pt.append(snnipetAnormal_pt)

I_2 = []
for videoAnormal in range(len(F_a_pt)):

    mean_delta_pt = F_a_pt[videoAnormal].mean() # (1/T) x sum(delta_t, t=1 to T)
    delta_tilde_pt = F_a_pt[videoAnormal] - mean_delta_pt # delta_tilde_t = *delta_t - mean_delta, [T]; ecuation (1)

    R_t = torch.zeros_like(delta_tilde_pt)
    for i in range(1, len(delta_tilde_pt)):
        R_t[i] = max(0, R_t[i-1] + delta_tilde_pt[i]) # Acumulative sum and ReLU activation function
                                                   # Ecuation (2),  [T],
    RT_tilde = R_t / R_t.max() # Ecuation (3),  [T[0:1]] 
    
    RT_tilde_mean = RT_tilde.mean()
    
    I_2_j = torch.where(RT_tilde < RT_tilde_mean, torch.zeros_like(RT_tilde), 1)
    
    I_2.append(I_2_j)