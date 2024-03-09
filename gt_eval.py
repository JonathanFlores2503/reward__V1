# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def gtByFrames (gt_Database, gt_Numframes):
    gt_Numframes = np.genfromtxt(gt_Numframes, delimiter=',')
    df = pd.read_csv(gt_Database, sep='\s+', header=None)
    df = df.values
    gt_video = []
    gt_nameVideo = []
    for numberVideo in range(df.shape[0]):
        gt_frame = []

        if df[numberVideo,0][:6] != "Normal":
            gt_nameVideo.append(df[numberVideo,0][:-4])
            for i in range(int(gt_Numframes[numberVideo])):
                if df[numberVideo,4] == -1:
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
                    
            print(numberVideo, df[numberVideo,0][:-4], len(gt_frame))
            gt_video.append(gt_frame)
    return gt_video, gt_nameVideo

# gt_Database = "E:/UCF-Crime/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt"
# gt_Numframes = "num_frames.csv"
# gt_anomaly, gt_anomalyVideoName = gtByFrames (gt_Database, gt_Numframes)