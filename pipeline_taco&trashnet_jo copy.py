import serial
import threading
import time
import torch
import numpy as np
import cv2
from datetime import datetime
from torchvision import models, transforms
from queue import Queue, Empty
from PIL import Image
from scipy.optimize import minimize
from ultralytics import YOLO

# Load the model outside the loop
taco2trashnet = np.loadtxt("tacototrashnet.txt", dtype=int) 
trashnet_cat = {0: "glass",
                1: "paper",
                2: "cardboard",
                3: "plastic",
                4: "metal",
                5: "trash"}

net = models.detection.ssdlite320_mobilenet_v3_large(weigths_backbone="DEFAULT", num_classes=60)
net.load_state_dict(torch.load("taco&trashnet_full_50.pt", map_location=torch.device('cpu')))
net.eval()
done = False#variable which tells the Raspi if the Arduino is done moving
while done != True: 
    plastic_count = 0
    glass_count = 0
    cardboard_count = 0
    paper_count = 0
    metal_count = 0
    trash_count = 0
    target_angle = 90  # Initialize target_angle
    

    def read_serial(ser, queue, stop_event, object_detected_event):
        time.sleep(0.2)
        global target_angle  # Declare target_angle as global
        time.sleep(1.5)
        ser.write(str(target_angle).encode('utf-8'))
        print(f"Sent first target angle to Arduino, value: {target_angle}")
        try:
            while not stop_event.is_set():
                serialdata = ser.readline().decode().strip()
                queue.put(serialdata)
                print(serialdata)  # Print received data
                if "Target reached" in serialdata:
                    if target_angle == 90:
                        target_angle = 180
                    elif target_angle == 180:
                        target_angle = 0
                    elif target_angle == 0:
                        target_angle = 90
                    ser.write(str(target_angle).encode('utf-8'))
                    print(f"Sent new target angle to Arduino: {target_angle}")  # Send target angle to Arduino
                if object_detected_event.is_set():
                    print("Serial port left open")
                    ser.write("Object has been spotted".encode('utf-8'))
                    time.sleep(0.5) #give time to the Arduino to decode the message
                    distance_servo_pairs = []
                    while len(distance_servo_pairs) < 30:
                        serialdata = ser.readline().decode().strip()  # Read new data
                        if "Distance:" in serialdata:
                            distance = float(serialdata.split("Distance: ")[1])
                            servo_values = []
                            for _ in range(4):  # Assuming 3 servo values per distance
                                serialdata = ser.readline().decode().strip()
                                servo_values.append(serialdata)
                            distance_servo_pairs.append((distance, servo_values))
                            timestamp = datetime.now()  # Get current timestamp
                            lista.append((timestamp, distance))  # Append timestamp along with distance
                            print(f"{timestamp}: Distance: {distance}, Servo values: {servo_values}")
                    # Extract distances and servo values separately
                    for distance, servo_values in distance_servo_pairs:
                        current_servo_angles.extend(servo_values)
                    #print(current_servo_angles)
                    #print(lista)
                    print("Serial port remains open")
                    return  # Exit the thread if object detected
        except serial.SerialException as ex:
            print(ex)  # Handle unplugged Arduino or other serial errors here

    def check_keyboard_input(stop_event):
        input("Press 'q' to stop the program:\n")
        stop_event.set()

    def initialize_camera():
        # Try to open the camera
        cap = cv2.VideoCapture(0)
        
        # Check if the camera is opened successfully
        if not cap.isOpened():
            raise RuntimeError("Failed to open the camera. Please ensure that the camera is connected and accessible.")

        # Set camera properties (if needed)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
        cap.set(cv2.CAP_PROP_FPS, 26)

        # Convert color space from BGR to RGB
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)

        return cap

    def object_detection(cap, object_detected_event):
        global plastic_count 
        global glass_count 
        global cardboard_count 
        global paper_count 
        global metal_count
        global trash_count 
        with torch.no_grad():
            while True:
                ret, image = cap.read()
                if not ret:
                    raise RuntimeError("Failed to read frame from the camera.")
                print('Camera set up')
                image = image[:, :, [2, 1, 0]]
                permuted = image.astype(np.uint8).copy()
                input_tensor = torch.tensor((image/255).astype(np.float32)).permute(2,1,0)
                input_batch = input_tensor.unsqueeze(0)
                output = net(input_batch)

                boxes = output[0]["boxes"]
                scores = output[0]["scores"]
                labels = output[0]["labels"]
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break # Show the frame continuously
                print("Imshow Set up")
                
                # Process detected objects
                for i in range(len(scores)):
                    #if object is detected
                    xmin, ymin, xmax, ymax = boxes[i]
                    if scores[i] > 0.95 and (xmax-xmin < 280 or ymax-ymin < 280):
                        print("Object spotted!")
                        xmin, ymin, xmax, ymax = boxes[i]
                        cv2.rectangle(permuted, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                        cv2.putText(permuted, trashnet_cat[taco2trashnet[(labels[i+1].item())]], (int((xmin+xmax)/2), int((ymin+ymax)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255))
                        print(trashnet_cat[taco2trashnet[(labels[i+1].item())]])
                        if trashnet_cat[taco2trashnet[(labels[i+1].item())]] == 'plastic':
                            plastic_count += 1
                        elif trashnet_cat[taco2trashnet[(labels[i+1].item())]] == 'glass':
                            glass_count += 1
                        elif trashnet_cat[taco2trashnet[(labels[i+1].item())]] == 'metal':
                            metal_count += 1
                        elif trashnet_cat[taco2trashnet[(labels[i+1].item())]] == 'paper':
                            paper_count += 1
                        elif trashnet_cat[taco2trashnet[(labels[i+1].item())]] == 'trash':
                            trash_count += 1
                        else:
                            cardboard_count += 1
                        print(xmin, xmax)
                        object_detected_event.set()
                        # Set the event when object is detected

                # Display the frame
                cv2.imshow('Frame', permuted)

                # Check for 'q' press to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # Define the serial port and baud rate
    serial_port  = 'COM3'  # Update this with your Arduino's serial port

    # Create a lock for serial port access
    lock = threading.Lock()

    # Initialize the camera
    try:
        cap = initialize_camera()
    except Exception as e:
        print(f"Error: {e}")
        exit()

    # List to store distance measurements
    lista = []
    current_servo_angles = []
    # Create an event to signal object detection
    object_detected_event = threading.Event()

    # Start a thread for object detection
    
    try:
        # Start a thread for reading serial data
        serial_thread = threading.Thread(target=read_serial, args=(serial.Serial(serial_port, 9600, timeout=1), Queue(), threading.Event(), object_detected_event))
        serial_thread.start()

        object_detection_thread = threading.Thread(target=object_detection, args=(cap, object_detected_event))
        object_detection_thread.start()


    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Stopping threads...")

    # Release the camera capture object
    
    # Wait for the serial thread to finish
    serial_thread.join()

    distances = [pair[1] for pair in lista]
    print(distances)
    # Calculate the mean of distances
    mean = sum(distances) / len(distances)

    print(f"The object is approximately {mean} cm deep")
    print(current_servo_angles)


    # Extracting servo values
    t1 = current_servo_angles[0] 
    t2 = current_servo_angles[1]  
    t3 = current_servo_angles[2]  
    t5 = current_servo_angles[3]
    # Printing extracted values
    print("Servo 1 value:", t1)
    print("Servo 2 value:", t2)
    print("Servo 3 value:", t3)
    print("Servo 5 value:", t5)
    t1 = int(t1)
    t2 = int(t2)
    t3 = int(t3)
    t5 = int(t5)

    #convert angles to radians for usage
    t1 = np.deg2rad(t1)
    t2 = np.deg2rad(t2)
    t3 = np.deg2rad(t3)
    t5 = np.deg2rad(t5)

    t = t1,t2,t3,t5

    def pos(t):
        t1,t2,t3,t5 = t
        t3 = np.pi/2-t3
        t5 = t5-0.87266463 #calibration
        #print(t3)

        Q1 = np.zeros(3)
        Q2 = Q1 + np.array([np.cos(t1)*1.5, np.sin(t1)*1.5,8])
        Q3 = Q2 + 13.3 * np.array([np.cos(t2)*np.cos(t1), np.cos(t2)*np.sin(t1), np.sin(t2)])
        Q4 = Q3 + 11 * np.array([np.cos(t2+t3)*np.cos(t1), np.cos(t2+t3)*np.sin(t1), np.sin(t2+t3)])
        Q5 = Q4 + 11 * np.array([np.cos(t2+t3+t5)*np.cos(t1),np.cos(t2+t3+t5)*np.sin(t1), np.sin(t2+t3+t5)])
        UH = Q4 + 6.5 * np.array([np.cos(t2+t3+t5)*np.cos(t1),np.cos(t2+t3+t5)*np.sin(t1), np.sin(t2+t3+t5)])
        Uhnormal = UH + mean* np.array([np.cos(t2+t3+t5-np.pi/2)*np.cos(t1),np.cos(t2+t3+t5-np.pi/2)*np.sin(t1), np.sin(t2+t3+t5-np.pi/2)])

        

        return Q1,Q2,Q3,Q4,Q5, UH, Uhnormal

    Q1,Q2,Q3,Q4,Q5, UH, Uhnormal = pos(t)
    print(Q5)
    X = np.array([UH[0]-2,UH[1]-2,Q5[2]-mean]) # approximation of objects coordinates
    print(X)
    t1 = np.arctan2(X[1],X[0])

    if X[1]<0:
        t1 += np.pi

    def dist(t):
        t2,t3,t5 = t
        Q1,Q2,Q3,Q4,Q5, UH, Uhnormal = pos([t1,t2,t3,t5])
        return np.sqrt(np.sum((Q5-X)**2))

    if X[1] < 0:
        res = minimize(dist,
            [0,0,0],
            bounds = [(0,np.pi),(0,np.pi/2), (-0.34906585, 1.74532925)],
            method = 'trust-constr')
    else:
        res = minimize(dist,
            [0,0,0],
            bounds = [(0,np.pi),(np.pi/2, np.pi), (-0.34906585, 1.74532925)],
            method = 'trust-constr')

    t1 = np.rad2deg(t1)
    print(t1)
    t1 = int(t1)
    if t1 < 0:  
        t1 = 180 + t1

    t2 = (res.x[0]/np.pi)*180
    print(t2)
    t2 = int(t2)

    t3 = (res.x[1]/np.pi)*180
    print(t3)
    t3 = int(t3)

    t5 = (res.x[2]/np.pi)*180
    t5 = t5 + 40
    print(t5)
    t5 = int(t5)

    print(t1,t2,t3,t5)

    new_serial_queue = Queue()
        # Convert angles to strings and add newline characters
    def read_serial_output(ser, new_serial_queue):
            while True:
                line = ser.readline().decode('utf-8').rstrip()
                new_serial_queue.put(line)
                print(line)
                if line == "Done moving":
                    done = True
                    break
            return

        # Convert angles to strings and add newline characters
    t1 = int(t1)
    t2 = int(t2)
    t3 = int(t3)
    t5 = int(t5)

    t1 = str(t1)
    t2 = str(t2)
    t3 = str(t3)
    t5 = str(t5)
    print(t1)
    print(t2)
    print(t3)
    print(t5)
    t1 += "\n"
    t2 += "\n"
    t3 += "\n"
    t5 += "\n"
    # Open serial port
    if __name__ == '__main__':
        ser = serial.Serial('COM3', 9600)
    if not ser.isOpen():
        ser.open()
    print('COM3 Open:', ser.isOpen())
    ser.reset_input_buffer()
    time.sleep(1)
    # Create a thread to read serial output
    serial_thread = threading.Thread(target=read_serial_output, args=(ser, new_serial_queue))
    serial_thread.start()

    # Continuously send servo angles
    while True:
        ser.write("Send\n".encode('utf-8'))
        print("Message sent")
        time.sleep(1)
        ser.write(t1.encode('utf-8'))
        print(f"{t1} angle sent")
        ser.write(t2.encode('utf-8'))
        print(f"{t2} angle sent")
        ser.write(t3.encode('utf-8'))
        print(f"{t3} angle sent")
        ser.write(t5.encode('utf-8'))
        print(f"{t5} angle sent")
        time.sleep(1)
        if "Moved servo5" in list(new_serial_queue.queue):
            #done = True
            break
    done = True
