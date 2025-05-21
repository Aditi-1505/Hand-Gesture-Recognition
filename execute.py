import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize the camera capture
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Use only 1 hand

# Load the model 
try:
    classifier = Classifier("C:/Users/Aditi/Desktop/gesture/keras_model.h5", 
                            "C:/Users/Aditi/Desktop/gesture/labels.txt")
except Exception as e:
    print("Error loading model or labels:", e)
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Image preprocessing parameters
offset = 20
imgSize = 300
counter = 0

# Define the label names 
labels = ["Call","Fist","Hello","Okay", "Love You","Peace","Thank you","Thumbs Down","Thumbs Up"]  

while True:
    success, img = cap.read()
    
    # Check if the camera frame was successfully captured
    if not success:
        print("Failed to capture image")
        break
    
    imgOutput = img.copy()

    # Detect hands in the image
    hands, img = detector.findHands(img)
    
    # If hands are detected
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # Get bounding box of the hand

        # Create a white canvas for resizing the hand image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the detected hand from the original image
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        
        if imgCrop.size == 0:  # Check if the cropped image is empty
            print("Empty crop detected, skipping frame")
            continue
        
        imgCropShape = imgCrop.shape

        # Calculate aspect ratio of the hand (height vs width)
        aspectRatio = h / w

        # Check aspect ratio to determine how to resize the cropped hand
        if aspectRatio > 1:
            # If the height is greater than the width, adjust based on height
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)  # Add padding if necessary
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            # If the width is greater than the height, adjust based on width
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal)) 
            hGap = math.ceil((imgSize - hCal) / 2) # Centering the image
            imgWhite[hGap:hCal + hGap, :] = imgResize  # Placing the image on canvas

        # Prediction using the classifier
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        
        # Draw a rectangle around the detected hand and label it
        cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x + w + offset, y - offset + 60), (128, 0, 128), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)  
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y+h + offset), (128, 0, 128), 4)

        # Show the cropped and resized images
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)
    
    # Show the final output with the hand detected and classified
    cv2.imshow('Image', imgOutput)

    # Exit loop if 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
