# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import csv
 

featuresI3DDir_path = "I3D_features/5fps_padding/Train" # Path of precomputed features I3D
keyWord = 'feature'
csv_Na_dir = "Distance"
csv_output_dir = "Features"     
dirI_1 = "TraningData"
nameFeatures_csv = "features.csv"
nameLabels_csv = "labales.csv"


"""Page 4 Set generation F_n, F_a"""
featuresI3D_classes = os.listdir(featuresI3DDir_path)
F_n = []
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
            print(featureName[:-4] + ": " + str(features[keyWord].shape))
            os.makedirs(featurePath_csv, exist_ok=True)
            F_a = features[keyWord].squeeze(axis=0)
            # print(F_a.shape)
            #----Comment if you previously saved this data and want to speed up the calculation process----
            with open(os.path.join(featurePath_csv, (featureName[:-4] + "_features.csv") ), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(F_a)

F_n = np.concatenate(F_n, axis=1)
print("F_n dimension: ", F_n.shape) # F_n = (1, N = [N_n*T], 1024)

#----Uncomment if you want to save the normal features (not necessary for this model)----
# F_n_list = F_n[0].tolist()
# with open(csv_output_dir + "/F_n_Features.csv", 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(F_n_list)
#----Uncomment if you want to save the normal features (not necessary for this model)----

"""Distance layer"""
print("Start Distance Calculation: Distance layer")
F_n = torch.from_numpy(F_n).to('cuda').squeeze(0)
F_n = F_n.to(dtype=torch.float32)
for classNameFiles in featuresI3D_classes:
    if classNameFiles != "Normal":
        print(classNameFiles)
        csv_Na_name = os.path.join(csv_Na_dir,classNameFiles)
        os.makedirs(csv_Na_name, exist_ok=True)
        featuresI3D_csv = os.listdir(os.path.join(csv_output_dir, classNameFiles))
        for N_a in featuresI3D_csv:
            f_a = np.genfromtxt(os.path.join(csv_output_dir, classNameFiles, N_a), delimiter=',')
            f_a = torch.from_numpy(f_a).to('cuda') # [T x P]
            f_a = f_a.to(dtype=torch.float32)
            if f_a.dim() == 1:
                f_a = f_a.unsqueeze(0)
            # diff_squared = (f_a[:, None, :] - F_n[None, :, :]).pow(2) # Calculates squared difference between each pair of points.
            # distanceLayer = diff_squared.sum(dim=2).sqrt() # Sums squared differences and calculates square root to get Euclidean 
            deltaDistances = torch.cdist(f_a, F_n, p=2) # Delta = [delta_ti]_{t=1}^{T, N} {i=1} = [T x N] 
            deltaDistances = deltaDistances.cpu().numpy()
            
            assert(f_a.size(0) == deltaDistances.shape[0]) 
            
            #----Comment if you previously saved this data and want to speed up the calculation process----
            with open(os.path.join(csv_Na_name, (N_a[:-13] + "_distance.csv")), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(deltaDistances)
            
            print(N_a + ":" + str(deltaDistances.shape))
            
"""Page 4 Steps: Sorting layer, average pooling layer, Smoothing and Initial selection"""
k = 20
thr_lambda = 0.8
I_1_features = []

print("Start the steps: Sorting layer, Average pooling layer, Smoothing and Initial selection")
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
                
            print("Second step: Sorting layer")
            sorted_delta, _ = csvDistance_Na.sort(dim=1, descending=False) # Sorter layer [T x N]
            print("Third step: Average pooling layer")
            averageSortedDelta = sorted_delta[:, :k].mean(dim=1) # Average pooling layer KNN, [T]
            
            print("Fourth step: Smoothing")
            # Normalization of distance values around zero
            mean_delta = averageSortedDelta.mean() # (1/T) x sum(delta_t, t=1 to T)
            delta_tilde = averageSortedDelta - mean_delta # delta_tilde_t = *delta_t - mean_delta, [T]; ecuation (1)
            
            # delta_tilde = delta_tilde.cpu().numpy()
            
            # Acumulative sum operation and a ReLU activation function
            D_t = torch.zeros_like(delta_tilde)
            for i in range(1, len(delta_tilde)):
                D_t[i] = max(0, D_t[i-1] + delta_tilde[i]) # Acumulative sum and ReLU activation function
                                                           # Ecuation (2),  [T]
                                                           
            print("Fifth step: Smoothing")
            # Normalization of D_t by its largest value,
            testerDt = D_t / D_t.max() # Ecuation (3),  [T[0:1]] 
            
            # Selection of segments greater than lambda
            D_lambda_thresholded = torch.where(testerDt < thr_lambda, torch.zeros_like(testerDt), testerDt)     
            
            print("Sixth step: Initial selection I_1")
            # Final Selection:
            f_a = np.genfromtxt(os.path.join(csv_output_dir, classNameFiles, (distanceNa[:-13] + "_features.csv")), delimiter=',')
            print(distanceNa[:-13], ":", len(f_a))
            if f_a.ndim == 1:
                f_a = [f_a.tolist()]
            elif f_a.ndim == 0:
                f_a = [[f_a]]
            else:
                f_a = f_a.tolist()
                
            countSnnipet = 0
            I_1_labels = []
            features_to_write = []
            for index in range(D_lambda_thresholded.numel()):
                if D_lambda_thresholded[index] != 0:
                    print("Anormal index: ", index)
                    I_1_labels.append(1)
                    countSnnipet += 1
                    features_to_write.append(f_a[index][:])
                    
            #----Comment if you previously saved this data and want to speed up the calculation process----
            with open(os.path.join(dirI_1, nameFeatures_csv), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(features_to_write)
                
            #----Comment if you previously saved this data and want to speed up the calculation process----
            with open(os.path.join(dirI_1, nameLabels_csv), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows([[label] for label in I_1_labels])
            print("Total number of anormal's snnipets: " + str(countSnnipet) + "\n")
            
    I_1_features_database = np.genfromtxt(os.path.join(dirI_1, nameFeatures_csv), delimiter=',')
    I_1_labels_database  = np.genfromtxt(os.path.join(dirI_1, nameLabels_csv), delimiter=',')
    print("I_1_features: ", I_1_features_database.shape)
    print("I_1_labels: ", I_1_labels_database.shape)
    assert(I_1_features_database.shape[0] == I_1_labels_database.shape[0])

""" Nominal segments as label  0"""
F_n = F_n.cpu().numpy()

#----Comment if you previously saved this data and want to speed up the calculation process----
file_is_empty = not os.path.exists(os.path.join(dirI_1, nameFeatures_csv)) or os.stat(os.path.join(dirI_1, "features.csv")).st_size == 0
with open(os.path.join(dirI_1, nameFeatures_csv), 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if not file_is_empty:
        csvfile.write('\n')
    for row in F_n:
        writer.writerow(row)

fnLables = np.zeros(F_n.shape[0], dtype="float32")

#----Comment if you previously saved this data and want to speed up the calculation process----
with open(os.path.join(dirI_1, nameLabels_csv), 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for label in fnLables:
        writer.writerow([label])

I_1_features_database = np.genfromtxt(os.path.join(dirI_1, nameFeatures_csv), delimiter=',')
I_1_labels_database  = np.genfromtxt(os.path.join(dirI_1, nameLabels_csv), delimiter=',')
print("I_1_features: ", I_1_features_database.shape)
print("I_1_labels: ", I_1_labels_database.shape)
assert(I_1_features_database.shape[0] == I_1_labels_database.shape[0])
