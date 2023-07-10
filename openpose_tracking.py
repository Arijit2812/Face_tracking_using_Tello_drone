import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import math

from djitellopy import tello
import time

#Uncomment for running the drone stream
# me = tello.Tello()
# me.connect()
# print(me.get_battery())
# #
# me.streamon()
# me.takeoff()
# me.send_rc_control(0, 0, 25, 0)
# time.sleep(1.0)

w, h = 360, 240
fbRange = [150, 300]
pid = [0.4, 0.4, 0]
pError = 0

def detect_pose(opWrapper, params, frame):
    # Pass image to OpenPose
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Extract pose keypoints
    if datum.poseKeypoints is not None:
        pose_keypoints = datum.poseKeypoints[0]
        pose_len = len(pose_keypoints)
        x_coords = []
        y_coords = []

        # Loop through each pose keypoint and append x and y coordinates to lists
        for i in range(pose_len):
            x_coords.append(pose_keypoints[i][0])
            y_coords.append(pose_keypoints[i][1])
            cv2.circle(frame, (int(pose_keypoints[i][0]), int(pose_keypoints[i][1])), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

        # Get keypoint positions for nose, left and right eyes, and left and right ears
        nose = datum.poseKeypoints[0][0]
        left_eye = datum.poseKeypoints[0][15]
        right_eye = datum.poseKeypoints[0][16]
        left_ear = datum.poseKeypoints[0][17]
        right_ear = datum.poseKeypoints[0][18]

        # Calculate the center point of the face
        center_x = int((nose[0] + left_eye[0] + right_eye[0] + left_ear[0] + right_ear[0]) / 5)
        center_y = int((nose[1] + left_eye[1] + right_eye[1] + left_ear[1] + right_ear[1]) / 5)

        # Get keypoint positions for left and right shoulders
        left_shoulder = datum.poseKeypoints[0][5]
        right_shoulder = datum.poseKeypoints[0][2]
        right_wrist = datum.poseKeypoints[0][4]

        # Calculate distance between shoulders
        distance = math.sqrt(
            (left_shoulder[0] - right_shoulder[0]) ** 2 + (left_shoulder[1] - right_shoulder[1]) ** 2)

        # Calculate the coordinates of the top-left and bottom-right corners of the bounding box
        x1 = int(center_x - distance//2)
        y1 = int(center_y - distance//2)
        x2 = int(center_x + distance//2)
        y2 = int(center_y + distance//2)

        # Draw a bounding box around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Calculate the area of the bounding box
        area = (x2 - x1) * (y2 - y1)

        # Convert center_x and center_y to a NumPy array
        center = np.array([center_x, center_y])

        # Get the grayscale image of the face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = gray[y1:y2, x1:x2]

        # Apply a binary threshold to include pixels 127 to 255
        _, face_binary = cv2.threshold(face, 0, 255, cv2.THRESH_BINARY)

        # Calculate the number of non-zero pixels in the face
        num_nonzero = cv2.countNonZero(face_binary)

        # Calculate the total number of pixels in the face region
        total_pixels = area

        # Calculate the visibility score
        visibility = num_nonzero / total_pixels * 100

        return [center, area, visibility]

    else:
        return [[0, 0], 0, 0]


def track_face(info, w, pid, pError):
    area = info[1]
    x, y = info[0]
    fb = 0

    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))

    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20

    if x == 0:
        speed = 0
        error = 0

    #print(speed, fb)
    #me.send_rc_control(0, fb, 0, speed) #Uncomment for running the drone stream
    return error


if __name__ == "__main__":
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../bin/python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../x64/Release;' +  dir_path + '/../bin;'
            import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument('--net_resolution', type=str, default='-1x128', help='OpenPose net resolution')
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../models/"
        params["net_resolution"] = "-1x128"
        params["disable_multi_thread"] = True

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Process Webcam Stream
        cap = cv2.VideoCapture(0) #Comment out for running the drone stream

        # Initialize FPS calculation variables
        prev_time = 0
        fps = 0

        avg_time =[]

        #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #out = cv2.VideoWriter('openpose_output.avi', fourcc, 10.0, (360, 240))


        while True:
            # Start the timer
            start_time = cv2.getTickCount()
            success, frame = cap.read() #Comment out for running the drone stream
            #frame = me.get_frame_read().frame #Uncomment for running the drone stream
            frame = cv2.resize(frame, (w, h))

            # Calculate FPS
            curr_time = cv2.getTickCount()
            time_elapsed = (curr_time - prev_time) / cv2.getTickFrequency()
            fps = 1 / time_elapsed
            prev_time = curr_time

            keypoints = detect_pose(opWrapper, params, frame)
            if keypoints is not None:
                center, area, visibility = keypoints
                pError = track_face([center, area], w, pid, pError)

            #frame = draw_pose(frame, keypoints)
            # End the timer and calculate the elapsed time
            end_time = cv2.getTickCount()
            time_taken = (end_time - start_time) / cv2.getTickFrequency()

            avg_time.append(time_taken)
            if len(avg_time) != 0:
                ex_time = sum(avg_time) / len(avg_time)
                print("Average time", ex_time)

            print (time_taken)
            print (center, area, visibility)
            #print (area)
            #print (visibility)
            #out.write(frame)  # Write the frame to the output file.
            cv2.imshow("Openpose_Output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                #me.land()
                break

        # Cleanup
        #cap.release()
        #cv2.destroyAllWindows()

    except Exception as e:
        print(e)
        sys.exit(-1)



