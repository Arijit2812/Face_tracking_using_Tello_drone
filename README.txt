# COSC428 Project

## Real-Time Face Detection and Tracking Using Tello Drone

This project proposes a method to compare some of the techniques of face detection e.g. Haar cascade, object detection using neural network, keypoint and pose estimation. The comparison is done on the baseline of receiving video from the webcam and outputting the movement commands to the terminal. Thereafter, face detection and tracking is done using a Tello drone on a computer with limited specifications. Drone motion control uses a PID control algorithm that can work well on a comprehensive system with good results.

### Installation for Windows 10 OS:
- Install CUDA https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
- Install OpenCV with CUDA support from source https://techzizou.com/setup-opencv-dnn-cuda-module-for-windows/
- Install Openpose https://github.com/CMU-Perceptual-Computing-Lab/openpose
- Install mediapipe https://github.com/google/mediapipe/blob/master/docs/getting_started/python.md
- Install djitellopy library
- Download yolov3 weights https://pjreddie.com/darknet/yolo/
- Download haarcascade weights https://github.com/kipr/opencv/tree/master/data/haarcascades



### Usage
- Change filepaths in all the relevant python files to point to the downloaded weights 
- Run haarcascade_tracking.py 
- Run yolov3_tracking.py
- Run mediapipe_tracking.py
- Run openpose_tracking.py
- Compare the different models
- Connect the drone to the laptop through wifi
- Run demo.py to fly the drone