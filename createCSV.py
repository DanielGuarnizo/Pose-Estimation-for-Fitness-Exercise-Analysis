# CAPTURE LANDMARKS AND EXPORT TO CSV

import mediapipe as mp
import cv2
import numpy

import csv
import os
import numpy as np
from matplotlib import pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# TO CREATE THE FIRST ROW OF THE CSV FILE
landmarks = ['class']
for val in range(1, 33+1): # 33 landmarks in total
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

with open('/Users/danielguarnizo/Desktop/Computer Vision /Notes/CSV_files/coords_DL_C.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

def export_landmark(results, action):
    try:
        keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        print(keypoints)
        keypoints.insert(0,action)

        with open('/Users/danielguarnizo/Desktop/Computer Vision /Notes/CSV_files/coords_DL_C.csv', mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(keypoints)
    except Exception as e:
        pass


cap = cv2.VideoCapture('Videos/CorrectDeadlift_45f.mp4')
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        # Recolor Feed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = pose.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        

        k = cv2.waitKey(1)
        if k == 117: # which letter does it correspond to??
            export_landmark(results, 'up')
        #if k == ord('u'):
            #export_landmark(results, 'up')
        if k == 100:
            export_landmark(results, 'down')
        #if k == ord('d'):
            #export_landmark(results, 'down')

        cv2.imshow('Raw Webcam Feed', image)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

