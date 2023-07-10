import cv2
import numpy as np
from djitellopy import tello
import time

#Uncomment for running the drone stream
# me = tello.Tello()
# me.connect()
# print(me.get_battery())
#
# me.streamon()
# me.takeoff()
# me.send_rc_control(0, 0, 25, 0)
# time.sleep(2.0)

w, h = 360, 240
fbRange = [6200, 6800]
pid = [0.4, 0.4, 0]
pError = 0

def find_face(img):
    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    # Apply thresholding to convert the grayscale image to a binary image (for calculating visibility)
    #_, imgThresh = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY)
    _, imgThresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate the number of white pixels in the binary image
    numWhitePixels = np.count_nonzero(imgThresh == 255)

    # Calculate the total number of pixels in the image
    numPixels = imgThresh.shape[0] * imgThresh.shape[1]

    # Calculate the percentage of visibility
    visibilityPercentage = numWhitePixels / numPixels * 100

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        #return img, [myFaceListC[i], myFaceListArea[i]]
        return img, [myFaceListC[i], myFaceListArea[i]], visibilityPercentage
    else:
        #return img, [[0, 0], 0]
        return img, [[0, 0], 0], visibilityPercentage


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

    print(speed, fb)
    #me.send_rc_control(0, fb, 0, speed) #Uncomment for running the drone stream
    return error


cap = cv2.VideoCapture(0) #Comment out for running the drone stream
# Initialize FPS calculation variables
prev_time = 0
fps = 0

avg_time =[]

#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter('haar_output.avi', fourcc, 10.0, (w, h))

while True:

    start_time = cv2.getTickCount()
    _, img = cap.read() #Comment out for running the drone stream
    #img = me.get_frame_read().frame #Uncomment for running the drone stream
    img = cv2.resize(img, (w, h))

    # Calculate FPS
    curr_time = cv2.getTickCount()
    time_elapsed = (curr_time - prev_time) / cv2.getTickFrequency()
    fps = 1 / time_elapsed
    prev_time = curr_time

    img, info, visibility = find_face(img)

    #img, info = find_face(img)

    end_time = cv2.getTickCount()

    pError = track_face(info, w, pid, pError)

    # End the timer and calculate the elapsed time


    time_taken = (end_time - start_time) / cv2.getTickFrequency()
    avg_time.append(time_taken)
    if len(avg_time) != 0:
        ex_time = sum(avg_time) / len(avg_time)
        print ("Average time", ex_time)
    # Add the processing time to the frame
    cv2.putText(img, f"Processing Time: {time_taken:.5f} seconds", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                 (0, 0, 255), 2)
    print("Center", info[0], "Area", info[1], visibility)

    # Display FPS on output frame

    #img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    #out.write(img)  # Write the frame to the output file.
    #cv2.putText(img, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Haarcascade_Output", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        #me.land
        break
