#!/usr/bin/env python3

import rospy
import cv2
import csv
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from deepface import DeepFace

class FaceRecognizer:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        self.csv_file = '/home/pikachu/face_data.csv'
        rospy.loginfo("FaceRecognizer node initialized and subscribed to /usb_cam/image_raw")

        # Check if CSV file exists, if not create it and add headers
        if not os.path.isfile(self.csv_file):
            with open(self.csv_file, mode='w') as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Favorite Drink"])

    def image_callback(self, data):
        rospy.loginfo("Received an image!")
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            rospy.loginfo("Image converted to OpenCV format")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Use DeepFace for face detection and recognition
        try:
            rospy.loginfo("Analyzing the image with DeepFace")
            analysis = DeepFace.analyze(cv_image, actions=['age', 'gender', 'race', 'emotion'])
            for face in analysis['instances']:
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                cv2.rectangle(cv_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                rospy.loginfo(f"Detected face: {face}")

                # Prompt the user for name and favorite drink
                name = input("Enter the name of the detected person: ")
                favorite_drink = input(f"Enter {name}'s favorite drink: ")

                # Write the data to CSV file
                with open(self.csv_file, mode='a') as file:
                    writer = csv.writer(file)
                    writer.writerow([name, favorite_drink])

        except Exception as e:
            rospy.logerr(f"DeepFace Error: {e}")

        # Display the image
        cv2.imshow("Face Recognition", cv_image)
        cv2.waitKey(3)

def main():
    rospy.init_node('face_recognizer', anonymous=True)
    fr = FaceRecognizer()
    rospy.spin()

if __name__ == '__main__':
    main()
