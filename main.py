import cv2
from ultralytics import YOLO
import threading
from flask import Flask
from flask_socketio import SocketIO

def object_tracking(model, wanted_items, cap):
        while True:
            ret, frame = cap.read() #Captures the frames 

            #If a frame is not returned
            if not ret:
                print("Unable to recieve the frame")
                break

            #The results of using the YOLO model to track objects in the frame 
            results = model.track(frame, verbose=False) 

            #Loops through results to see if the detected object is one of the wanted items 
            for r in results[0].boxes:
                if model.names[r.cls.item()] in wanted_items:
                    print("Wanted item detected: ", model.names[r.cls.item()])
                

            annotated_frame = results[0].plot() #Gets the list the labels and draws bounding boxes on the objects. 

            cv2.imshow('Webcam Feed', annotated_frame) #Displays the webcam feed

            #Ends the camera feed if 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                print("Exiting...")
                break 

#Load Yolov8
model = YOLO("yolov8n.pt")

#A list of all the items wanted to be detected
wanted_items = ["person", "car", "truck", "dog"]

#Setting up the flask app
app = Flask(__name__)

@app.route("/")
def home_page():
    return "<h1> Testing Lad </h1>"

cap = cv2.VideoCapture(0) #Captures video at the webcam index 0

#If the computer is unable to open the webcam 
if not cap.isOpened(): 
    print("Unable to open camera.")

try:
    #Start the object tracking loop in a background thread
    threads = []
    threads.append(threading.Thread(target=object_tracking, args=(model, wanted_items, cap), daemon=True)) #daemon=True means that the thread dies when the program exits
    
    # Start each thread
    for t in threads:
        t.start()

    #Run the Flask app
    app.run(debug=False, use_reloader=False)

    # Wait for all threads to finish
    for t in threads:
        t.join()
    

finally:
    #Ends the camera feed 
    cap.release()
    cv2.destroyAllWindows()
    
    