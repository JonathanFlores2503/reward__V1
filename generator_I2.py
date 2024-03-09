# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 02:18:06 2024

@author: jonat
"""

import os
import numpy as np
import torch
import mlpArchitecture
import matplotlib.pyplot as plt
import gt_eval
import math
from sklearn import metrics
import cv2

"""Raw Features"""
dirRawFeatures_path = "FeaturesTest/normal" #Test features without padding
# dirRawFeatures_path = "FeaturesTest/padding" #Test features with padding
gt_Database = "Temporal_Anomaly_Annotation.txt"
pretraningModel_path = "Models/MLP_I1_v1.pth"
gt_Numframes = "num_frames.csv"
saveGraphs = "Test2"
dirVideo_path = "UCF-Crime Path" # Database path (videos) UCF-Crime
numVideo = 1 # [0-139]

"""Raw Features"""
featuresI3D_classes = os.listdir(dirRawFeatures_path)

"""Model"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mlpArchitecture.shallowNN().to(device)
model.load_state_dict(torch.load(pretraningModel_path, map_location=device))
model.eval()

"""I_2 """
I_2_labels = []
snippetSize = 16 * 6 # 16 = Number of frames in a snippet
                     # 6 = Uniform separation between 5 consecutive frames selected from the original rate of 30 fps
                     # 16 x 6 = 96 [Total frames analyzed by snippet]
                     # 3.2s analyzed by snippet.
for classesI3D in featuresI3D_classes:
    featuresOneClasse_path = os.path.join(dirRawFeatures_path, classesI3D)
    featuresOneClasseFiles = os.listdir(featuresOneClasse_path)
    for featureName in featuresOneClasseFiles:
        feature_path = os.path.join(featuresOneClasse_path, featureName)
        F_n = np.genfromtxt(feature_path, delimiter=',')
        F_n = torch.from_numpy(F_n).to('cuda') 
        F_n = F_n.to(dtype=torch.float32)
        # print(featureName, ":", F_n.size())
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
        
""" Obtaining GT and metrics"""
gt_anomalyFrames, gt_anomalyVideoName = gt_eval.gtByFrames (gt_Database, gt_Numframes)
auc = 0
for i in range(len(gt_anomalyFrames)):
    if len(gt_anomalyFrames[i]) <  len(I_2_labels[i]):
        fpr, tpr, thresholds = metrics.roc_curve(np.array(gt_anomalyFrames[i]), np.array(I_2_labels[i][:len(gt_anomalyFrames[i])]), pos_label=1)
        if math.isnan(metrics.auc(fpr, tpr)):
            auc += 0
            print(i, gt_anomalyVideoName[i], len(gt_anomalyFrames[i]), len(I_2_labels[i]), 0)
        else:
            auc += metrics.auc(fpr, tpr)
            print(i, gt_anomalyVideoName[i], len(gt_anomalyFrames[i]), len(I_2_labels[i]), metrics.auc(fpr, tpr))
    else:
        fpr, tpr, thresholds = metrics.roc_curve(np.array(gt_anomalyFrames[i][:len(I_2_labels[i])]), np.array(I_2_labels[i]), pos_label=1)
        if math.isnan(metrics.auc(fpr, tpr)):
            auc += 0
            print(i, gt_anomalyVideoName[i], len(gt_anomalyFrames[i]), len(I_2_labels[i]), 0)
        else:
            auc += metrics.auc(fpr, tpr)
            print(i, gt_anomalyVideoName[i], len(gt_anomalyFrames[i]), len(I_2_labels[i]), metrics.auc(fpr, tpr))

        
assert(len(gt_anomalyFrames) == len(I_2_labels))
print(auc/len(gt_anomalyFrames))

#----Uncomment if you want to save the graphs----
# for i in range(len(I_2_labels)):
#     VideoNumberTest = i
#     plt.figure(figsize=(10, 5))
#     plt.plot(I_2_labels[i][:len(gt_anomalyFrames[i])], label='I_2')
#     plt.plot(gt_anomalyFrames[VideoNumberTest], label='GT')
#     if VideoNumberTest > 58:
#         VideoNumberTest += 150
#     plt.title(("Comparison I_2 vs GT" + " [" + str(gt_anomalyVideoName[i]+ "]")))
#     plt.xlabel('Frames')
#     plt.ylabel('Labels')
#     plt.legend()
#     # plt.savefig(("Graphs/" + saveGraphs +"/" + str(gt_anomalyVideoName[i]) + ".jpg"), bbox_inches='tight')
#     plt.show()

"""Test Video"""
#----Uncomment if you want to take the visual tests by video----
# video_number = gt_anomalyVideoName[numVideo]
# classVideo = gt_anomalyVideoName[numVideo][:-8]
# videoTest = dirVideo_path + "/" + classVideo + "/" + video_number + ".mp4"
# cap = cv2.VideoCapture(videoTest)
# if not cap.isOpened():
#     print("Error! Can't open the video file.")
#     exit()
# frame_count = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     cv2.putText(frame, ("# frame: " + str(frame_count)), (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA )
#     if gt_anomalyFrames[numVideo][frame_count] == 1:
#         cv2.putText(frame, ("GT: Anomaly detected"), (5,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA )
#     if I_2_labels[numVideo][frame_count] == 1:
#         cv2.putText(frame, ("I_2 prediction: Anomaly detected"), (5,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA )
#     cv2.imshow('frame', frame)
#     frame_count += 1
#     if len(I_2_labels[numVideo]) <= frame_count:
#         break
#     if cv2.waitKey(1) == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
    