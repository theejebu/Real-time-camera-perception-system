import cv2
from ultralytics import YOLO
import threading
from flask import Flask
from flask_socketio import SocketIO
import queue

def object_tracking(model, wanted_items, cap):
        while True:
            ret, frame = cap.read() #Captures the frames 

            #If a frame is not returned
            if not ret:
                print("Unable to recieve the frame")
                break

            #The results of using the YOLO model to track objects in the frame 
            results = model.track(frame, verbose=False, classes=wanted_ids) 

            #Loops through results to see if the detected object is one of the wanted items 
            for r in results[0].boxes:
                if model.names[r.cls.item()] in wanted_items:
                    print("Wanted item detected: ", model.names[r.cls.item()])
                    detection_data = {"class":model.names[r.cls.item()], "id":r.id.item(), "confidence":r.conf.item()} #Make a dictionary for the queue
                    detection_queue.put(detection_data) #Add it into the Queue

            annotated_frame = results[0].plot() #Gets the list the labels and draws bounding boxes on the objects

            cv2.imshow('Webcam Feed', annotated_frame) #Displays the webcam feed

            #Ends the camera feed if 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                print("Exiting...")
                break 

#Load Yolov8
model = YOLO("yolov8n.pt")

#A list of all the items  and ids wanted to be detected. The items will later be converted into ids.
wanted_items = ["person", "car", "truck", "dog"]
wanted_ids = []

#Convert the wanted items into ids 
for id, name in model.names.items():
    if name in wanted_items:
        wanted_ids.append(id)

#Make a queue
detection_queue = queue.Queue()

#Setting up the flask app
app = Flask(__name__)
#Link SocketIO to the flask app
socketio = SocketIO(app)

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
    
    #Start each thread
    for t in threads:
        t.start()

    #Run the Flask app
    socketio.run(app, debug=False, use_reloader=False)

    #Wait for all threads to finish
    for t in threads:
        t.join()
    

finally:
    #Ends the camera feed 
    cap.release()
    cv2.destroyAllWindows()
    
    