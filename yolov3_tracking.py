import cv2
import numpy as np
from djitellopy import tello
import time
import os

#Uncomment for running the drone stream
# me = tello.Tello()
# me.connect()
# print(me.get_battery())
# #
# me.streamon()
# me.takeoff()
# me.send_rc_control(0, 0, 25, 0)
# time.sleep(1.0)

w, h = 320, 320
fbRange = [40000, 42000]
pid = [0.4, 0.4, 0]
pError = 0

def find_face(frame):

    net = cv2.dnn.readNetFromDarknet('Resources/cfg/yolov3.cfg', 'Resources/yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    with open('Resources/coco.names.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Resize frame to input size of YOLOv3 model
    height, width, channels = frame.shape
    input_size = (320, 320)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, input_size, swapRB=True, crop=False)

    # Pass blob through YOLOv3 model
    net.setInput(blob)
    outputs = net.forward(output_layers)

    myFaceListC = []
    myFaceListArea = []

    class_ids = []
    confidences = []
    boxes = []
    #print (outputs)
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #if confidence > 0.5 and class_id in [0, 39, 67]:  # Detect multiple classes
            if confidence > 0.5 and class_id == 0:  # Only detect faces (class ID 0)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    for i in indices:
        i = i
        x, y, w, h = boxes[i]
        cx = x + w // 2
        cy = y + w // 2
        area = w * h  # calculate area of bounding box
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}, area: {area}"
        #cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        # calculate visibility percentage
        #visibility = max(0, min(100, (myFaceListArea[i] / (frame.shape[0] * frame.shape[1])) * 100))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        w = h = int(np.sqrt(area))
        top, left = max(cy - h // 2, 0), max(cx - w // 2, 0)
        bottom, right = min(cy + h // 2, frame.shape[0]), min(cx + w // 2, frame.shape[1])
        face = gray[top:bottom, left:right]
        _, face_binary = cv2.threshold(face, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        visibility = cv2.countNonZero(face_binary) / area
        visibility >= 0.5, visibility

        # add visibility to label
        #label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}, area: {area}, visibility: {visibility_percentage}%"
        #label = f"visibility: {visibility * 100}%"
        #cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #print (visibility_percentage)
        return frame, [myFaceListC[i], myFaceListArea[i]]
    else:
        return frame, [[0, 0], 0]


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
        fb = -2
    elif area < fbRange[0] and area != 0:
        fb = 2

    if x == 0:
        speed = 0
        error = 0

    #print(speed, fb)
    #me.send_rc_control(0, fb, 0, speed) #Uncomment for running the drone stream
    return error

# Initialize webcam
cap = cv2.VideoCapture(0) #Comment out for running the drone stream

# Initialize FPS calculation variables
prev_time = 0
fps = 0

avg_time = []
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter('yolo_output.avi', fourcc, 2.0, (360, 240))

while True:
    start_time = cv2.getTickCount()
    # Read frame from webcam
    _, frame = cap.read() #Comment out for running the drone stream
    #frame = me.get_frame_read().frame #Uncomment for running the drone stream
    frame = cv2.resize(frame, (w, h))

    # Calculate FPS
    curr_time = cv2.getTickCount()
    time_elapsed = (curr_time - prev_time) / cv2.getTickFrequency()
    fps = 1 / time_elapsed
    prev_time = curr_time

    frame, info = find_face(frame)
    end_time = cv2.getTickCount()
    pError = track_face(info, w, pid, pError)
    # End the timer and calculate the elapsed time

    time_taken = (end_time - start_time) / cv2.getTickFrequency()

    avg_time.append(time_taken)
    if len(avg_time) != 0:
        ex_time = sum(avg_time) / len(avg_time)
        print("Average time", ex_time)
    print("Center", info[0], "Area", info[1])
    #print("Area", info[1])
    print (time_taken)

    #frame = cv2.resize(frame, (360, 240), interpolation=cv2.INTER_AREA)
    #out.write(frame)  # Write the frame to the output file.

    # Display FPS on output frame
    #cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Yolo_Output", frame)

    # Break if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #me.land()
        break

# Release webcam and close window
#cap.release()
#cv2.destroyAllWindows()
