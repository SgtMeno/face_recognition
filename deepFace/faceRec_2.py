#!/home/pikachu/catkin_ws/src/face_recognition_pkg/deepFace/.deep/bin/python3

import rospy
import cv2
import face_recognition
import csv
import os
import time
from collections import Counter
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerRequest
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# ROS Node and Topic Names
Node = "face_recognition_node"
FaceOutputTopic = "user_data"
SpeechResponseTopic = 'speech'
SpeechTriggerService = 'start_speech_recognition'
TextTriggerService = 'tts_request'
Camera = '/usb_cam/image_raw'
# Path to the CSV file
csv_file = "/home/pikachu/catkin_ws/src/face_recognition_pkg/deepFace/faceData/faceData.csv"

# Load known faces and their labels
known_face_encodings = []
known_face_names = []
known_user_drinks = {}

# Load data from CSV if it exists
if os.path.exists(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            name = row[0]
            drink = row[1]
            encoding = eval(row[2])
            known_face_names.append(name)
            known_user_drinks[name] = drink
            known_face_encodings.append(encoding)

# Global variables
awaiting_name = False
awaiting_drink = False
pending_name = None
pending_encoding = None
bridge = CvBridge()

# Detection window variables
window_duration = 3  # Detection window duration in seconds
detection_window = []

def save_user_data(name, drink, encoding):
    # Save user data to CSV
    with open(csv_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([name, drink, encoding])

def activate_tts(message):
    try:
        rospy.set_param('/tts_message', message)  # Passing the message
        rospy.wait_for_service(TextTriggerService)
        tts_service = rospy.ServiceProxy(TextTriggerService, Trigger)
        tts_service(TriggerRequest())  # Trigger TTS with the message
    except rospy.ServiceException as e:
        rospy.logerr(f"TTS Service call failed: {e}")

def speech_callback(data):
    global awaiting_name, awaiting_drink, pending_name, pending_encoding
    rospy.loginfo(f"Speech Callback Triggered - Awaiting Name: {awaiting_name}, Awaiting Drink: {awaiting_drink}")
    if awaiting_name:
        pending_name = data.data.strip()
        rospy.loginfo(f"Received name: {pending_name}")
        awaiting_name = False
        awaiting_drink = True
        activate_tts("What is your favorite drink?")

    elif awaiting_drink:
        drink = data.data.strip()
        rospy.loginfo(f"Received favorite drink: {drink}")
        activate_tts(f"Greetings {pending_name}, Your favorite drink is {drink}.")
        save_user_data(pending_name, drink, list(pending_encoding))
        rospy.loginfo(f"New User Added: {pending_name} - Favorite Drink: {drink}")

        awaiting_drink = False
        pending_name = None
        pending_encoding = None
        rospy.set_param('/tts_message', "")  # Resetting the name

def process_image(image_msg):
    global awaiting_name, awaiting_drink, pending_encoding

    # Skip processing if awaiting user input
    if awaiting_name or awaiting_drink:
        return

    try:
        # Convert ROS image message to OpenCV image
        frame = bridge.imgmsg_to_cv2(image_msg, "bgr8")
        rospy.loginfo("Successfully converted ROS image to OpenCV format.")
    except CvBridgeError as e:
        rospy.logerr(f"Failed to convert image: {e}")
        return

    start_time = time.time()
    while time.time() - start_time < window_duration:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings:
            closest_face_encoding = face_encodings[0]

            matches = face_recognition.compare_faces(known_face_encodings, closest_face_encoding, tolerance=0.5)
            name = "Unknown"
            drink = "N/A"

            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]
                drink = known_user_drinks[name]
                rospy.loginfo(f"Recognized: {name} - Favorite Drink: {drink}")

                # Activate TTS to announce the recognized person
                activate_tts(f"Hello {name}, your favorite drink is {drink}.")
            else:
                pending_encoding = closest_face_encoding
                rospy.loginfo("Requesting user input...")
                activate_tts("Hello there, nice to meet you. May I know your name?")
                awaiting_name = True
                rospy.set_param('/awaiting_name', True)
                break

            detection_window.append(name)

        # Draw rectangles and labels on the frame
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}", (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"{drink}", (left, bottom + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if detection_window:
        most_common_face = Counter(detection_window).most_common(1)[0][0]
        drink = known_user_drinks.get(most_common_face, "N/A")
        pub.publish(f"Name: {most_common_face}, Drink: {drink}")
        rospy.loginfo(f"List of detections: {detection_window}")
        rospy.loginfo(f"Published: Name: {most_common_face}, Drink: {drink}")

    detection_window.clear()

def detect_and_recognize_faces():
    rospy.init_node(Node, anonymous=True)
    rospy.Subscriber(SpeechResponseTopic, String, speech_callback)
    rospy.Subscriber(Camera, Image, process_image)
    rospy.loginfo("Face Recognition Node Started")
    rospy.spin()  # Keep the node running and processing callbacks
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detect_and_recognize_faces()
    except rospy.ROSInterruptException:
        pass
