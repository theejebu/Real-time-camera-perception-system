#Imports
import cv2
from ultralytics import YOLO
import threading
from flask import Flask, Response, render_template 
from flask_socketio import SocketIO
import queue
import numpy as np
import datetime

def object_tracking(model, wanted_items, cap):
        global current_frame
        while True:

            ret, frame = cap.read() #Captures the frames 

            #If a frame is not returned
            if not ret:
                print("Unable to recieve the frame")
                break

            #The results of using the YOLO model to track objects in the frame 
            results = model.track(frame, verbose=False, classes=wanted_ids, persist=True) 

            if results[0].boxes.id is not None:
                # Extract IDs and Classes
                object_ids = results[0].boxes.id.int().cpu().tolist()
                object_confidence = results[0].boxes.conf.cpu().tolist()
                object_class = results[0].boxes.cls.int().cpu().tolist()

                for id, conf, cls in zip(object_ids, object_confidence, object_class):
                    # Only send if ID is new and in wanted_items
                    if id not in sent_ids:
                        class_name = model.names[cls]
                        current_time = datetime.datetime.now().strftime("%c")
                        
                        detection_data = {
                            "class": class_name, 
                            "time": current_time, 
                            "confidence": round(conf, 2), 
                            "id": id
                        }

                        detection_queue.put(detection_data) #Add the data to the queue
                        sent_ids.add(id) #Add the unique ID to the IDs that are already sent

                annotated_frame = results[0].plot() #Gets the list the labels and draws bounding boxes on the objects

                #Locks the frame so it can safely be written to
                with lock:
                    current_frame = annotated_frame 

def generate_frames():
    while True:
            if current_frame is not None:

                #Lock so the current frame can safely be read from
                with lock:
                    frame = current_frame

                successful_encoding, encoded_frame = cv2.imencode(".jpg", frame) #Encodes the frame into a jpg 
                encoded_frame = encoded_frame.tobytes() #Convert the image into bytes
                
                #If the encoding doesn't work 
                if not successful_encoding:
                    print("Unsucessful encoding ")
                    break
                
                #Yield the frame into the MJPEG format
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n\r\n')

def browser_transmission():
    while True:
        recieved_data = detection_queue.get()
        socketio.emit("Detection", recieved_data) #Send the data from the detection to the browser

# Track IDs already transmitted to avoid duplicates
sent_ids = set()

#Load Yolov8
model = YOLO("yolov8n.pt")

#A list of all the items  and ids wanted to be detected. The items will later be converted into ids
wanted_items = ["person", "car", "truck", "dog"]
wanted_ids = []

#Convert the wanted items into ids 
for id, name in model.names.items():
    if name in wanted_items:
        wanted_ids.append(id)

#Lock for the threads so they don't read and write a variable at the same time
lock = threading.Lock()

#Make a queue
detection_queue = queue.Queue()

#Setting up the flask app
app = Flask(__name__)

#Link SocketIO to the flask app
socketio = SocketIO(app)

#Route for the home page
@app.route("/")
def home_page():
    return render_template('index.html')

#Route for the video feed
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

cap = cv2.VideoCapture(0) #Captures video at the webcam index 0
current_frame = None #Declare a global variable for the current frame 

#If the computer is unable to open the webcam 
if not cap.isOpened(): 
    print("Unable to open camera.")

try:
    #Start the object tracking loop in a background thread
    threads = []

    threads.append(threading.Thread(target=object_tracking, args=(model, wanted_items, cap), daemon=True)) #Thread to track the objects
    threads.append(threading.Thread(target=browser_transmission, daemon=True)) #Thread to send data to browser
    
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
    
    