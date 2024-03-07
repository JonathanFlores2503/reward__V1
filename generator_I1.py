# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 00:56:45 2024

@author: jonat
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import pandas as pd

#%% 
# Page 4 Set generation F_n, F_a
featuresI3DDir_path = "E:\\UCF-Crime\\I3D_Features\\Test"
nameModel = "rewardMLP"
featuresI3D_classes = os.listdir(featuresI3DDir_path)
F_n = []
keyWord = 'feature'
csv_Na_dir = "Distance"
csv_output_dir = "Features"     
dirI_1 = "TraningData"
nameFeatures_csv = "featuresTest.csv"
nameLabels_csv = "labalesTest.csv"
 
for classesI3D in featuresI3D_classes:
    featuresOneClasse_path = os.path.join(featuresI3DDir_path, classesI3D)
    featuresOneClasseFiles = os.listdir(featuresOneClasse_path)
    featurePath_csv = os.path.join(csv_output_dir, classesI3D)
    print(classesI3D)
    for featureName in featuresOneClasseFiles:
        feature_path = os.path.join(featuresOneClasse_path, featureName)
        # print(feature_path)  
        features = np.load(feature_path)
        if classesI3D == "Normal":
            F_n.append(features[keyWord])
        else:
            # print(featureName[:-4] + ": " + str(features[keyWord].shape))
            os.makedirs(featurePath_csv, exist_ok=True)
            F_a = features[keyWord].squeeze(axis=0)
            
            #----------------------comment----------------------
            with open(os.path.join(featurePath_csv, (featureName[:-4] + "_features.csv") ), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(F_a)

F_n = np.concatenate(F_n, axis=1)
print("F_n dimension: ", F_n.shape) # F_n = (1, N = [N_n*T], 1024)

#----------------------comment----------------------
# F_n_list = F_n[0].tolist()
# with open(csv_output_dir + "/F_n_Features.csv", 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(F_n_list)
    
print("Start Distance Calculation: Distance layer")
F_n = torch.from_numpy(F_n).to('cuda').squeeze(0)
F_n = F_n.to(dtype=torch.float32)

#If you just add Fn in the database, uncomment the nex line code
# 0/0
for classNameFiles in featuresI3D_classes:
    if classNameFiles != "Normal":
        print(classNameFiles)
        csv_Na_name = os.path.join(csv_Na_dir,classNameFiles)
        os.makedirs(csv_Na_name, exist_ok=True)
        featuresI3D_csv = os.listdir(os.path.join(csv_output_dir, classNameFiles))
        for N_a in featuresI3D_csv:
            f_a = np.genfromtxt(os.path.join(csv_output_dir, classNameFiles, N_a), delimiter=',')
            f_a = torch.from_numpy(f_a).to('cuda') #[T x P]
            f_a = f_a.to(dtype=torch.float32)
            if f_a.dim() == 1:
                f_a = f_a.unsqueeze(0)
            # diff_squared = (f_a[:, None, :] - F_n[None, :, :]).pow(2) # Calculates squared difference between each pair of points.
            # distanceLayer = diff_squared.sum(dim=2).sqrt() # Sums squared differences and calculates square root to get Euclidean 
            deltaDistances = torch.cdist(f_a, F_n, p=2) # Delta = [delta_ti]_{t=1}^{T, N} {i=1} = N_a * [T x N] 
            deltaDistances = deltaDistances.cpu().numpy()
            
            assert(f_a.size(0) == deltaDistances.shape[0]) 
            
            #----------------------comment----------------------
            with open(os.path.join(csv_Na_name, (N_a[:-13] + "_distance.csv")), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(deltaDistances)
            
            print(N_a + ":" + str(deltaDistances.shape))
    # deltaDistances.append(distanceLayer)
# 0/0
#%%    
# Page 4 Distance Calculation: Sorting layer, average pooling layer
k = 20
thr_lambda = 0.8
I_1_features = []

snippetSize = 16 * 5

print("Second step: Distance Calculation: Sorting layer, average pooling layer")
for classNameFiles in featuresI3D_classes:
    if classNameFiles != "Normal":
        fullDistancesList = os.listdir(os.path.join(csv_Na_dir, classNameFiles))
        for distanceNa in fullDistancesList:
            csvDistance_Na = np.genfromtxt(os.path.join(csv_Na_dir, classNameFiles,distanceNa), delimiter=',')
            csvDistance_Na = torch.from_numpy(csvDistance_Na).to('cuda') # N_a * [T x N] 
            print(distanceNa, ":", csvDistance_Na.size())

            if csvDistance_Na.dim() == 1:
                csvDistance_Na = csvDistance_Na.unsqueeze(0)
                dim_to_sort = 0  
            elif csvDistance_Na.size(0) == 1:
                dim_to_sort = 0
            else:
                dim_to_sort = 1

            sorted_delta, _ = csvDistance_Na.sort(dim=1, descending=False) # Sorter layer [T x N]
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
            f_a = np.genfromtxt(os.path.join(csv_output_dir, classNameFiles, (distanceNa[:-13] + "_features.csv")), delimiter=',')
            if f_a.ndim == 1:
                f_a = [f_a.tolist()]
            elif f_a.ndim == 0:
                f_a = [[f_a]]
            else:
                f_a = f_a.tolist()
                
            #----------------------comment----------------------
            with open(os.path.join(dirI_1, nameFeatures_csv), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(f_a)
            
            I_1_labels = []
            for index in  range(D_lambda_thresholded.numel()):
                if D_lambda_thresholded[index] != 0:
                    I_1_labels.append(1)
                    continue
                I_1_labels.append(0)
                
            I_1_preLabels = []
            for snippet in range(len(I_1_labels)):
                for i in range(snippetSize):
                    if I_1_labels[snippet] == 0:
                        I_1_preLabels.append(0)
                    else:
                        I_1_preLabels.append(1) 
            
            #----------------------comment----------------------
            with open(os.path.join(dirI_1, nameLabels_csv), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for label in I_1_labels:
                    writer.writerow([label])
            I_1_features.append(I_1_preLabels)



    I_1_features_database = np.genfromtxt(os.path.join(dirI_1, nameFeatures_csv), delimiter=',')
    I_1_labels_database  = np.genfromtxt(os.path.join(dirI_1, nameLabels_csv), delimiter=',')
    print("I_1_features: ", I_1_features_database.shape)
    print("I_1_labels: ", I_1_labels_database.shape)
    assert(I_1_features_database.shape[0] == I_1_labels_database.shape[0])
#%%
# F_n = F_n.cpu().numpy()
# file_is_empty = not os.path.exists(os.path.join(dirI_1, nameFeatures_csv)) or os.stat(os.path.join(dirI_1, "features.csv")).st_size == 0
# with open(os.path.join(dirI_1, nameFeatures_csv), 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     if not file_is_empty:
#         csvfile.write('\n')
#     for row in F_n:
#         writer.writerow(row)

# fnLables = np.zeros(F_n.shape[0], dtype="float32")
# with open(os.path.join(dirI_1, nameLabels_csv), 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for label in fnLables:
#         writer.writerow([label])

# I_1_features_database = np.genfromtxt(os.path.join(dirI_1, nameFeatures_csv), delimiter=',')
# I_1_labels_database  = np.genfromtxt(os.path.join(dirI_1, nameLabels_csv), delimiter=',')
# print("I_1_features: ", I_1_features_database.shape)
# print("I_1_labels: ", I_1_labels_database.shape)
# assert(I_1_features_database.shape[0] == I_1_labels_database.shape[0])

#%% Obtaining GT and metrics
import math
from sklearn import metrics

gt_Database = "E:/UCF-Crime/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt"
df = pd.read_csv(gt_Database, sep='\s+', header=None)
df = df.values
gt_video = []
auc = 0
countVideos = 0
for numberVideo in range(df.shape[0]):
    gt_frame = []
    if df[numberVideo,0][:6] != "Normal":
        
        for i in range(len(I_1_features[countVideos])):
            if df[numberVideo,4] == -1:
                # print("Una anomalia")
                if (i+1) >= df[numberVideo,2] and (i+1) <= df[numberVideo,3]:
                    gt_frame.append(1) 
                    continue                               
                else:
                    gt_frame.append(0) 
                    continue
            # print("Dos anomalia")
            if (i+1) >= df[numberVideo,2] and (i+1) <= df[numberVideo,3]:
                gt_frame.append(1) 
            elif (i+1) >= df[numberVideo,4] and (i+1) <= df[numberVideo,5]:
                gt_frame.append(1)
            else:
                gt_frame.append(0)
                
        fpr, tpr, thresholds = metrics.roc_curve(np.array(gt_frame), np.array(I_1_features[countVideos]), pos_label=1)
        
        if math.isnan(metrics.auc(fpr, tpr)):
            auc += 0
            print(countVideos, df[numberVideo,0], df[numberVideo,2:], len(I_1_features[countVideos]), 0)
        else:
            auc += metrics.auc(fpr, tpr)
            print(countVideos, df[numberVideo,0], df[numberVideo,2:], len(I_1_features[countVideos]), metrics.auc(fpr, tpr))
        gt_video.append(gt_frame)
        countVideos += 1

assert(len(gt_video) == len(I_1_features))
print(auc/countVideos)


VideoNumberTest = 118
plt.figure(figsize=(10, 5))
plt.plot(I_1_features[VideoNumberTest], label='I_1')
plt.plot(gt_video[VideoNumberTest], label='GT')
if VideoNumberTest > 58:
    VideoNumberTest += 150
plt.title(("Comparison I_1 vs GT" + "[" + str(df[VideoNumberTest,0] + "]")))
plt.xlabel('Snnipet')
plt.ylabel('Labels')
plt.legend()
plt.savefig(("Graphs/" + "Comparison I_1 vs GT.jpg"), bbox_inches='tight')
plt.show()
