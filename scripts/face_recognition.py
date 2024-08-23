#!/usr/bin/env python3

import csv
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import threading
import os

# Load the Haar Cascade classifier for face detection
cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

class FaceRecognizer:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        self.csv_file = '/home/pikachu/face_data.csv'
        self.face_id = 0
        self.detected_faces = set()
        
        rospy.loginfo("FaceRecognizer node initialized and subscribed to /usb_cam/image_raw")

        # Check if CSV file exists, if not create it and add headers
        if not os.path.isfile(self.csv_file):
            with open(self.csv_file, mode='w') as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Favorite Drink"])

        # Start a thread for user input
        self.input_thread = threading.Thread(target=self.get_user_input)
        self.input_thread.start()

    def get_user_input(self):
        while not rospy.is_shutdown():
            rospy.loginfo("Waiting for user input")
            name = input("Enter the name of the detected person: ")
            favorite_drink = input(f"Enter {name}'s favorite drink: ")
            # Save the user input
            with open(self.csv_file, mode='a') as file:
                writer = csv.writer(file)
                writer.writerow([name, favorite_drink])
            rospy.loginfo(f"Saved data: {name}, {favorite_drink}")

    def image_callback(self, data):
        rospy.loginfo("Received image data")
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            rospy.loginfo("Converted ROS image to OpenCV image")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        try:
            # Convert image to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            rospy.loginfo("Converted image to grayscale")

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            rospy.loginfo(f"Detected {len(faces)} faces")

            for (x, y, w, h) in faces:
                face_id = (x, y, w, h)
                # Check if this face is already detected
                if face_id not in self.detected_faces:
                    self.detected_faces.add(face_id)
                    # Draw rectangle around face
                    cv2.rectangle(cv_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    # Display face ID
                    cv2.putText(cv_image, f"Face {self.face_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    self.face_id += 1

            # Display the image with face detection
            cv2.imshow("Face Recognition", cv_image)
            cv2.waitKey(1)
            rospy.loginfo("Displayed image with face detection")

        except Exception as e:
            rospy.logerr(f"Cascade Error: {e}")

def main():
    rospy.init_node('face_recognizer', anonymous=True)
    fr = FaceRecognizer()
    rospy.loginfo("Face recognition node started")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
