# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 02:18:06 2024

@author: jonat
"""

import os
import numpy as np
import torch
import mlpArchitecture
import csv
import matplotlib.pyplot as plt
import pandas as pd

# featuresCSV_path = "TraningData/features.csv"
# I_2_labels_csv_path = "TraningData/labelsI2.csv"
#%% Raw Features
dirRawFeatures_path = "FeaturesTest" #Test features
featuresI3D_classes = os.listdir(dirRawFeatures_path)
#%% Model
pretraningModel_path = "Models/MLP_I1_full.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mlpArchitecture.shallowNN().to(device)
model.load_state_dict(torch.load(pretraningModel_path, map_location=device))
model.eval()
#%% I_2 
I_2_labels = []
snippetSize = 16 * 5
for classesI3D in featuresI3D_classes:
    featuresOneClasse_path = os.path.join(dirRawFeatures_path, classesI3D)
    featuresOneClasseFiles = os.listdir(featuresOneClasse_path)
    for featureName in featuresOneClasseFiles:
        feature_path = os.path.join(featuresOneClasse_path, featureName)
        F_n = np.genfromtxt(feature_path, delimiter=',')
        F_n = torch.from_numpy(F_n).to('cuda') 
        F_n = F_n.to(dtype=torch.float32)
        print(featureName, ":", F_n.size())
        with torch.no_grad():
            snnipetAnormal_pt = model(F_n) # [Tx1]

        # Normalization of distance values around zero
        mean_delta_pt = snnipetAnormal_pt.mean()# (1/T) x sum(delta_t, t=1 to T)
        delta_tilde_pt = snnipetAnormal_pt - mean_delta_pt # delta_tilde_t = *delta_t - mean_delta, [T]; ecuation (1)
        # Acumulative sum operation and a ReLU activation function
        R_t = torch.zeros_like(delta_tilde_pt)
        for i in range(1, len(delta_tilde_pt)):
            R_t[i] = max(0, R_t[i-1] + delta_tilde_pt[i]) # Acumulative sum and ReLU activation function
                                                         # Ecuation (2),  [T]
        # Normalization of D_t by its largest value,
        RT_tilde = R_t / R_t.max() # Ecuation (3),  [T[0:1]] 
        RT_tilde_mean = RT_tilde.mean()
        I_2_j = torch.where(RT_tilde < RT_tilde_mean, torch.zeros_like(RT_tilde), 1) # Ecuation (5),  [T[0:1]] 
        
        I_2_preLabels = []
        for snnipetLabel in I_2_j:
            snnipetLabel = int(snnipetLabel.cpu().numpy())
            for snippet in range(snippetSize):
                if snnipetLabel == 0:
                    I_2_preLabels.append(0)
                else:
                    I_2_preLabels.append(1)          
        I_2_labels.append(I_2_preLabels)
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
        
        for i in range(len(I_2_labels[countVideos])):
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
                
        fpr, tpr, thresholds = metrics.roc_curve(np.array(gt_frame), np.array(I_2_labels[countVideos]), pos_label=1)
        
        if math.isnan(metrics.auc(fpr, tpr)):
            auc += 0
            print(countVideos, df[numberVideo,0], df[numberVideo,2:], len(I_2_labels[countVideos]), 0)
        else:
            auc += metrics.auc(fpr, tpr)
            print(countVideos, df[numberVideo,0], df[numberVideo,2:], len(I_2_labels[countVideos]), metrics.auc(fpr, tpr))
        gt_video.append(gt_frame)
        countVideos += 1

assert(len(gt_video) == len(I_2_labels))
print(auc/countVideos)


VideoNumberTest = 118
plt.figure(figsize=(10, 5))
plt.plot(I_2_labels[VideoNumberTest], label='I_2')
plt.plot(gt_video[VideoNumberTest], label='GT')
if VideoNumberTest > 58:
    VideoNumberTest += 150
plt.title(("Comparison I_2 vs GT" + "[" + str(df[VideoNumberTest,0] + "]")))
plt.xlabel('Snnipet')
plt.ylabel('Labels')
plt.legend()
plt.savefig(("Graphs/" + "Comparison I_2 vs GT.jpg"), bbox_inches='tight')
plt.show()
    