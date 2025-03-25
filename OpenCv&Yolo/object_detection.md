**Dog and Object Detection Sample Code**

#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4


for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for i in indices:
    try:
        box = boxes[i]
    except:
        i = i[0]
        box = boxes[i]
    
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

cv2.imshow("object detection", image)
cv2.waitKey()
    
cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()


**Moving Object Detection**
What is Background Subtraction?
The first step of moving object detection is Background subtraction. Using a static camera is a common and widely used technique for generating a foreground mask (namely, a binary image containing the pixels belonging to moving objects in the scene).

As the name suggests, Background Subtraction calculates the foreground mask, performing a subtraction between the current frame and a background model containing the static part of the scene or, more generally, everything that can be considered background given the characteristics of the observed scene.

Background modeling consists of two main steps:

Background Initialization.
Background Update.
In the first step, an initial model of the background is computed, while in the second, the model is updated to adapt to possible changes in the scene. Background Estimation can also be applied to motion-tracking applications such as traffic analysis, people detection, etc. This article on background estimation for motion tracking will undoubtedly help you gain a better understanding.

Contour Detection in OpenCV
Contours can be explained simply as a curve joining all the continuous points (along the boundary), having the same color or intensity. The contours are useful for shape analysis, object detection, and recognition. OpenCV makes it easy to find and draw contours in images. 

It provides two simple functions:  findContours() and drawContours()

Also, it has two different algorithms for contour detection: CHAIN_APPROX_SIMPLE and CHAIN_APPROX_NONE

Setting Up the Environment
Before proceeding to the moving-object detection code, we must set up our environment. So you need to have Python and a preferred IDE on your computer, and that’s it. You are ready to go!

Required Software and Libraries
As we discussed previously, we need only OpenCV to install as the main driver of the code, and we will install Gradio to build a web app out of it as we always believe in practical and real-world approaches. 

Installing OpenCV and Gradio
We need to install those libraries into our Python environment. We suggest you to create a virtual environment and install all the dependencies for better version control:

! pip install opencv-python gradio


Importing all the Libraries:

import cv2
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt


Implementing Background Subtraction

cap = cv2.VideoCapture(vid_path)
backSub = cv2.createBackgroundSubtractorMOG2()
if not cap.isOpened():
    print("Error opening video file")
    while cap.isOpened():
        # Capture frame-by-frame
          ret, frame = cap.read()
          if ret:
            # Apply background subtraction
            fg_mask = backSub.apply(frame)


For background subtraction, we created another object using cv2.createBackgroundSubtractorMOG2. Further, we check whether our object cap is properly working or not. After that, we will start capturing all the video frames one by one using cap.read() method. Now, we will apply background subtraction to each video frame using backSub.apply() method.
Detecting and Drawing Contours:

# Find contours
contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(contours)
frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
# Display the resulting frame
cv2.imshow('Frame_final', frame_ct)


As we can see, all the moving pixels between two consecutive frames are detected. This includes cars, shadows, and some tiny white dots. Have you wondered why these tiny white dots are also present? Nothing is moving in those places, right!

When dealing with real-world scenarios, various environmental factors can affect CCTV footage. This may include weather conditions such as high winds.

Detecting and Drawing Contours:


# Find contours
contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(contours)
frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
# Display the resulting frame
cv2.imshow('Frame_final', frame_ct)

After background subtraction, we now have a clear binary mask of our ROI (Region of Interest). So, we will proceed to the contour detection. The above code block outlines the process. 

For this, we must pass the foreground mask (fg_mask) as input to the cv2.findContours functions to find all the possible contour points. This will happen for all the detected objects in the current frame. After that, we will pass all the contour points to cv2.drawContours function to draw outlines for each detected contour.

As you can see, drawing all the contours also includes a lot of unnecessary ones. These small, noisy contours will affect the results adversely further on. We address this issue in the next section.

Improving Contour Detection with  Image Thresholding and Morphological Operations
 The noisy contours appear due to the movement of the shadows and camera. We have to remove all the noise to get a clear frame with the detected contour of our ROI (car).

We perform 3 different steps to solve the problem. They are:

Thresholding
Erosion and Dilation
Contour Filtration


Thresholding:

# apply global threshold to remove shadows
 retval, mask_thresh = cv2.threshold( fg_mask, 180, 255, cv2.THRESH_BINARY)
For thresholding, we will use cv2.threshold function and pass all the foreground masks as input. We set the threshold value to 180 and the max value to 255, and it will simply replace all the values under 180 with 0 and remove all the shadows. And you can see the result that we obtained after the thresholding operation.

Thresholding in opencv moving object detection 
FIGURE 4 – Thresholding in OpenCV
Erosion and Dilation
After applying thresholding, now we have a binary image mask. This consists of a foreground mask of cars without any shadows. Still, the small white dots remain in the mask. The next step is removing the contours around these small dots:


# set the kernal
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Apply erosion
mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

To remove these small white dots now we will apply erosion and will dilate the image after for a better and wider mask. To do this, we will use cv2.morphologyEx and use cv2.MORPH_OPEN as MorphType and pass the thresholded mask mask_thresh into it. Let’s check out the results after the above two operations.

erosion and dilation in moving object detection
FIGURE 5 – Erosion and Dilation in OpenCV
We have obtained a much cleaner image now. The rest of the cleanup can be done by simply filtering the contours with small areas.


Filtering Contours:

min_contour_area = 500  # Define your minimum area threshold
large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

Filtering contours can sometimes be a trial-and-error approach. We find that, for our use case, a contour area of 500 works best. The following code block will filter out any contours less than 500 pixels. This approach retains the most prominent cars while filtering out the lane line.


Contour filtering in moving object detection:

frame_out = frame.copy()
for cnt in large_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
 
# Display the resulting frame
cv2.imshow('Frame_final', frame_out)

The next step of moving-object detection is drawing the bounding boxes around the remaining contours. This is where object detection takes place. To do this, first, we will extract all the box coordinates using the cv2.boundingRect function. Then we will draw bounding boxes for all detected objects for every video frame using cv2.rectangle. Let’s take a look at the results after drawing the bounding boxes.

Bounding Box Drawing in moving object detection opencv
FIGURE 7 – Drawing Bounding Box using OpenCV
Isn’t it amazing that we can achieve all of these with simple image processing and OpenCV?

Output video for moving object detection
FIGURE 8 – Final Output Video
Following is the output of an entire video for moving object detection with OpenCV.