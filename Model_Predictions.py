
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import mediapipe as mp
import cv2
import numpy as np

with open('/Users/danielguarnizo/Desktop/Computer Vision /Notes/deadliftt.pkl', 'rb') as f:
    model = pickle.load(f)



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

landmarks = ['class']
for val in range(1, 33+1): # 33 landmarks in total
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]


webcam = 1
cap = cv2.VideoCapture(webcam)
counter = 0
current_stage = ''
# frame_width = 600
# frame_height = 900
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, image = cap.read()
        print(image.shape)
        image = cv2.flip(image, 1)

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
        
        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns = landmarks[1:])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            
            if body_language_class == 'down' and body_language_prob[body_language_prob.argmax()] >= 0.7:
                current_stage = 'down'
            elif current_stage == 'down' and body_language_class == 'up' and body_language_prob[body_language_prob.argmax()] >= 0.7: # NON RIESCO A LEGGERE
                current_stage = 'up'
                counter +=1


            # cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Mediapipe Feed', 600, 900)
            
            # GET STATUT BOX
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

            # DISPLAY CLASS
            cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # DISPLAY PROBABILITY
            cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # DISPLAY COUNT
            cv2.putText(image, 'COUNT', (180, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (175,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        except Exception as e:
            print('VAFFANCULOOOO')

        cv2.imshow('Mediapipe Feed',image)
        # if webcam == 1:
        #     cv2.imshow('Mediapipe Feed', cv2.flip(image, 1))
        # else:
        #     cv2.imshow('Mediapipe Feed',cv2.flip(image, 1))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
cap.release()