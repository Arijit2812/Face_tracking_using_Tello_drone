import cv2
import mediapipe as mp
import math
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
# time.sleep(1.0)

w, h = 360, 240
fbRange = [80, 120]
pid = [0.4, 0.4, 0]
pError = 0


def find_face(frame):
    # Initialize face detection module
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()

    # Convert the image from BGR to RGB color space
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    results = face_detection.process(frame_rgb)

    myFaceListC = []
    myFaceListArea = []
    myFaceListVisibility = []

    # Draw bounding box around faces and calculate size
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            height, width, _ = frame.shape
            xmin = int(bbox.xmin * width)
            ymin = int(bbox.ymin * height)
            xmax = int((bbox.xmin + bbox.width) * width)
            ymax = int((bbox.ymin + bbox.height) * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            w = xmax - xmin
            h = ymax - ymin
            cx = xmin + w // 2
            cy = ymin + h // 2
            size = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)
            # Calculate visibility percentage
            visibility = detection.score[0] if detection.score[0] >= 0 else 0
            visibility_percentage = int(visibility * 100)

            #cv2.putText(frame, f"Visibility: {visibility_percentage}%", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #            0.5, (0, 255, 0), 2)
            #cv2.putText(frame, f"Size: {size:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            myFaceListC.append([cx, cy])
            myFaceListArea.append(size)
            myFaceListVisibility.append(visibility_percentage)

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return frame, [myFaceListC[i], myFaceListArea[i], myFaceListVisibility[i]]
    else:
        return frame, [[0, 0], 0, 0]

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

# Initialize video capture object
cap = cv2.VideoCapture(0) #Comment out for running the drone stream

# Initialize FPS calculation variables
prev_time = 0
fps = 0

avg_time =[]

#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter('mediapipe_output.avi', fourcc, 20.0, (w, h))

#while cap.isOpened():
while True:
    # Read a frame from the video capture object
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

    frame, info = find_face(frame)
    # End the timer and calculate the elapsed time
    end_time = cv2.getTickCount()
    pError = track_face(info, w, pid, pError)


    time_taken = (end_time - start_time) / cv2.getTickFrequency()

    avg_time.append(time_taken)
    if len(avg_time) != 0:
        ex_time = sum(avg_time) / len(avg_time)
        print("Average time", ex_time)

    # Add the processing time to the frame
    cv2.putText(frame, f"Processing Time: {time_taken:.5f} seconds", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)
    print("Center", info[0], "Area", info[1], info[2])
    #print("Area", info[1])
    #print (fps)

    #frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    #out.write(frame)  # Write the frame to the output file.
    # Display FPS on output frame
    #cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Display the resulting image
    cv2.imshow('Mediapipe_Output', frame)

    # Exit the program when 'q' is pressedq
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #me.streamoff()
        #me.land()
        break

# Release the video capture object and destroy all windows
#cap.release()
#cv2.destroyAllWindows()