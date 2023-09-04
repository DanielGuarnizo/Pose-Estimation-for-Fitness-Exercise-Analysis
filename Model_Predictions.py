import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import mediapipe as mp
import cv2
import numpy as np



def Make_Predictions(path_model, ups, downs, webcam):
    with open(path_model, 'rb') as f:
        model = pickle.load(f)

        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        
        cap = cv2.VideoCapture(webcam)
        counter = 0
        current_stage = ''

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            
            while cap.isOpened():
                ret, image = cap.read()
                # Mirrir image to mak easier the read on the window 
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
                    
                    if body_language_class in downs and body_language_prob[body_language_prob.argmax()] >= 0.1:
                        current_stage = body_language_class
                    elif current_stage in downs and body_language_class in ups and body_language_prob[body_language_prob.argmax()] >= 0.1: # NON RIESCO A LEGGERE
                        current_stage = body_language_class
                        counter +=1
                
                    cv2.rectangle(image, (0,0), (600, 120), (245, 117, 16), -1)

                    # DISPLAY CLASS
                    cv2.putText(image, 'CLASS', (190, 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0], (180,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4, cv2.LINE_AA)

                    # DISPLAY PROBABILITY
                    cv2.putText(image, 'PROB', (30, 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4, cv2.LINE_AA)

                    # DISPLAY COUNT
                    cv2.putText(image, 'COUNT', (480, 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2,cv2.LINE_AA)
                    cv2.putText(image, str(counter), (470,80), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),4,cv2.LINE_AA)


                except Exception as e:
                    print('VAFFANCULOOOO')

                cv2.imshow('Mediapipe Feed',image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
        cap.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

ups = ["up","up_close", "up_roll"]
downs = ["down", "down_close"]
model_path = "Models/Bench_rf.pkl"
webcam = 0
Make_Predictions(model_path, ups, downs, webcam)