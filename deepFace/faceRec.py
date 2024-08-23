#!/home/pikachu/catkin_ws/src/face_recognition_pkg/deepFace/.deep/bin/python3

import rospy
import cv2
import face_recognition
import csv
import os
import time
from std_msgs.msg import String
from collections import Counter

# ROS Node and Topic Names
Node = "face_recognition_node"
FaceOutputTopic = "user_data"
SpeechRequestTopic = 'gemini_request'
SpeechResponseTopic = 'speech'
SpeechTriggerTopic = 'start_speech_recognition'
TextTriggerTopic = 'tts_request'

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

def save_user_data(name, drink, encoding):
    # Save user data to CSV
    with open(csv_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([name, drink, encoding])

# State management
awaiting_name = False
awaiting_drink = False
pending_name = None
pending_encoding = None

def speech_callback(data):
    global awaiting_name, awaiting_drink, pending_name, pending_encoding

    if awaiting_name:
        # Capture the user name and trigger the drink prompt
        pending_name = data.data.strip()
        rospy.loginfo(f"Received name: {pending_name}")
        awaiting_name = False
        awaiting_drink = True
        # Trigger TTS to ask for the favorite drink
        rospy.Publisher(TextTriggerTopic, String, queue_size=10).publish(f"Good day {pending_name}, may I know your favorite drink?")

    elif awaiting_drink:
        # Capture the user drink and finalize data saving
        drink = data.data.strip()
        rospy.loginfo(f"Received favorite drink: {drink}")
        rospy.Publisher(TextTriggerTopic, String, queue_size=10).publish(f"Hello {pending_name}, you favourite drink is {drink}")
        # Save the new user data
        known_face_encodings.append(pending_encoding)
        known_face_names.append(pending_name)
        known_user_drinks[pending_name] = drink
        save_user_data(pending_name, drink, list(pending_encoding))
        rospy.loginfo(f"New User Added: {pending_name} - Favorite Drink: {drink}")

        # Reset the state and resume face detection
        awaiting_drink = False
        pending_name = None
        pending_encoding = None

def detect_and_recognize_faces():
    global awaiting_name, awaiting_drink, pending_encoding

    # Initialize ROS node
    rospy.init_node(Node, anonymous=True)
    pub = rospy.Publisher(FaceOutputTopic, String, queue_size=10)
    speech_pub = rospy.Publisher(TextTriggerTopic, String, queue_size=10)
    rospy.Subscriber(SpeechResponseTopic, String, speech_callback)
    rate = rospy.Rate(0.5) # 1 second
    # Capture video from webcam
    video_capture = cv2.VideoCapture(0)
    
    # Time-based sliding window variables
    detection_window = []
    window_duration = 5  # seconds
    consecutive_unknown_count = 0  # Track consecutive 'Unknown' detections
    unknown_threshold = 10  # Number of consecutive 'Unknown' detections before triggering user input

    while not rospy.is_shutdown():
        # Skip face detection if awaiting user input
        if awaiting_name or awaiting_drink:
            continue

        start_time = time.time()

        while time.time() - start_time < window_duration:
            ret, frame = video_capture.read()
            if not ret:
                rospy.logerr("Failed to capture frame from webcam")
                continue

            # Convert the frame to RGB (face_recognition expects RGB images)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect all face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if face_encodings:
                closest_face_encoding = face_encodings[0]  # Assuming the closest face is the first one detected

                # Check if the face is recognized
                matches = face_recognition.compare_faces(known_face_encodings, closest_face_encoding, tolerance=0.5)
                name = "Unknown"
                drink = "N/A"

                if True in matches:
                    # Find the recognized face
                    match_index = matches.index(True)
                    name = known_face_names[match_index]
                    drink = known_user_drinks[name]
                    rospy.loginfo(f"Recognized: {name} - Favorite Drink: {drink}")
                    # Reset the consecutive 'Unknown' counter if a known face is detected
                    consecutive_unknown_count = 0
                else:
                    # If 'Unknown' is detected, increment the counter
                    consecutive_unknown_count += 1

                    # If 'Unknown' has been detected for more than the threshold, prompt for user input
                    if consecutive_unknown_count >= unknown_threshold:
                        pending_encoding = closest_face_encoding
                        rospy.loginfo("Requesting user input...")
                        awaiting_name = True
                        speech_pub.publish("Please tell me your name:")
                        break

                # Track detections within the time window
                detection_window.append(name)

            # Draw a rectangle around the face and display the name and drink
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}", (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame, f"{drink}", (left, bottom + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Display the video feed
            cv2.imshow('Video', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Determine the most frequently detected face in the last 5 seconds
        if detection_window:
            most_common_face = Counter(detection_window).most_common(1)[0][0]  # Get the most common face name
            drink = known_user_drinks.get(most_common_face, "N/A")

            # Publish the data of the most frequently detected face
            pub.publish(f"Name: {most_common_face}, Drink: {drink}")
            rospy.loginfo(f"List of detections: {detection_window}")
            rospy.Publisher(TextTriggerTopic, String, queue_size=10).publish(f"Good day {most_common_face}, your favourite drink is {drink}, this is guest")
            rospy.loginfo(f"Published: Name: {most_common_face}, Drink: {drink}")


        # Clear the detection window for the next 5-second period
        detection_window.clear()

        rate.sleep()

    # Release the video capture object and close windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detect_and_recognize_faces()
    except rospy.ROSInterruptException:
        pass
