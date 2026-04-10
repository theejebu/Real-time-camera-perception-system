import cv2
from ultralytics import YOLO

#Load Yolov8
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0) #Captures video at the webcam index 0

#If the computer is unable to open the webcam 
if not cap.isOpened(): 
    print("Unable to open camera.")

try:
    while True:
        ret, frame = cap.read() #Captures the frames 

        #If a frame is not returned
        if not ret:
            print("Unable to recieve the frame")
            break

        results = model(frame) 

        annotated_frame = results[0].plot() #Gets the list the labels and draws bounding boxes on the objects. 

        cv2.imshow('Webcam Feed', annotated_frame) #Displays the webcam feed

        #Ends the camera feed if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            print("Exiting...")
            break 
        
finally:
    #Ends the camera feed 
    cap.release()
    cv2.destroyAllWindows()
    
    