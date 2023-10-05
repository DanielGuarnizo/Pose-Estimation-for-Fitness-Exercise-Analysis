# Abstract
This project presents a computer vision-based approach for analysing fitness exercises, specifically focusing on the squat, deadlift and bench press movements.
We leverage the MediaPipe framework to obtain landmark positions from video data and employ Scikit-learn to train models for the primary task of distinguishing between phases of an exercise repetition. These models serve as the foundation for achieving the main objectives of our system: repetition counting and exercise quality assessment.



# 1. INTRODUCTION
Pose Estimation is a computer vision task that involves detecting the position and orientation of an object or, as it pertains to our situation, of a person, namely Human Pose Estimation. For the latter, this is done by predicting the location of specific key-points, such as joints in the human body like shoulders, wrists, elbows, and other body parts. Nowadays, pose estimation has become a ubiquitous feature in a wide variety of applications. One typical example is physical fitness tracking and performance analysis, and our team has decided to focus precisely on this. Physical fitness is undoubtedly crucial for maintaining a healthy lifestyle, but effective exercise execution is even more fundamental for achieving fitness goals while minimizing the risk of injury. Our project implements a computer vision-based solution for analyzing three basic fitness exercises, the squat, the deadlift, and the bench press using pose estimation techniques through the MediaPipe framework and machine learning algorithms.

## 1.1. FUNCTIONAL OVERVIEW
The main idea for this project was to build an algorithm that could distinguish the different phases of an exercise execution so as to correctly count the repetitions and give some real-time suggestions to improve the form of the execution where needed.
Let’s take deadlifts as our example. Once we access our video camera through python using the OpenCV library, we can start working out doing our deadlifts. As we move in front of the camera, the information about our pose is extracted and used to categorize movements and consequentially display on screen (with the use of OpenCV) whatever we need. All of this is done in real time.
Figure 2 displays the screen during execution.

## CORRECT EXECUTION 

![Watch the video](https://github.com/DanielGuarnizo/Pose-Estimation-for-Fitness-Exercise-Analysis/assets/87019453/bba787a3-0d7d-4b2c-b8d8-5ba489984010)
## WRONG EXECUTION 
![Watch the video](https://github.com/DanielGuarnizo/Pose-Estimation-for-Fitness-Exercise-Analysis/assets/87019453/e44a566d-a40c-43da-a65a-3b336e294316)


# 2. METHODOLOGY
## 2.1. DATA COLLECTION AND PRE-PROCESSING
We first started by building a labeled dataset ourselves. We captured video footage of us individually performing squats, deadlifts, and bench presses in the most correct way possible. To use this data for subsequent analysis, we needed it to be in the form of relative positions of body parts in each phase of the exercise execution.
Continuing with the previous example of the deadlift, we aimed to distinguish the body pose at the top from the one at the bottom of the exercise repetition. To accomplish this the MediaPipe framework was employed to accurately extract the positions of landmarks throughout both the "UP" and "DOWN" phases of the exercise.
With a few lines of code, we were able to organize this data in the form of a CSV file. The first column of this file contains the classes, "UP" and "DOWN", while the rest of the columns denote the positions of the landmarks in these phases. (Figure 3)
We then repeated the same procedure to obtain data for the quality assessment of the exercise execution.
For the sake of consistency, we take up again the deadlift example. We wanted our model to distinguish between 4 possible classes of incorrect posture during a deadlift repetition: DOWN_LOW (hips too low), DOWN_ROLL (excessive downward bending of the back), UP_BACK (leaning too far backward), UP_ROLL (rounding of the back after completing the lift).


## 2.2. MODEL TRAINING
At this point, we needed to train some models, one for each of the three exercises, that could accurately distinguish between the different classes based on landmark positions.
We opted to use the open-source library Scikit- learn for Python as it provides a comprehensive set of machine learning algorithms and because of the offered utilities for evaluating these models through various metrics.
After loading the CSV file into a Pandas DataFrame and splitting it into train and test sets, we were uncertain which of the wide range of classifier algorithms we should opt for, so we decided to take advantage of how Scikit-learn simplifies the process of experimenting with the different ML algorithms.
We started by importing and initializing the classes of some classifiers we were interested in:
LogisticRegression, 
RidgeClassifier, 
RandomForestClassifier, 
GradientBoostingClassifier.
Then, we used a loop to iterate through a list of these models so as to train each of them on the common training data, make predictions on the same test data, and finally use some evaluation metrics to compare their performance. In detail, we used accuracy, precision, and recall. It was noteworthy that each one of the models performed exceptionally well, with scores that were very close to one another. However, we had to make a final decision and opted for the random forest classifier.
At this point, we had three (one for each exercise) pre-trained models crucial for recognizing different stages or poses during the fitness exercise.


# 3. CONCLUSION
In conclusion, the proposed system is able to identify exercise phases and effectively count repetitions. It can also distinguish between proper and improper execution of the exercises, demonstrating its potential to enhance fitness training and coaching. We have used MediaPipe’s Pose and leveraged the power of OpenCV to build a simple tool but we can further improve it by incorporating more advanced techniques, such as building a Human Action Recognition system modelling the temporal dependencies and relationships between frames system using a Recurrent Neural Networks (RNNs) and LSTM networks [2]. Another possible improvement could be expanding the scope to include additional exercises making it a more complete fitness tool.



