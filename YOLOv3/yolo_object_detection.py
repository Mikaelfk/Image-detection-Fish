import cv2
import csv
import numpy as np
import glob
import random
from intersectionOverUnion import dataList

index = 0

# Saves all the iou scores
iouScores = []

# Imports the csv file
with open("GroundTruth-numbered-only.csv", newline='') as csvfile:
   dataList = list(csv.reader(csvfile))

# y1, y2, x1, x2
for i in range(len(dataList)):
    dataTemp1 = dataList[i][1]
    dataTemp2 = dataList[i][2]
    dataList[i][1] = dataList[i][3]
    dataList[i][2] = dataTemp1
    dataList[i][3] = dataList[i][4]
    dataList[i][4] = dataTemp2

# Sorts the dataList after image name
dataList.sort()

# Computes the iou score between two rectangles
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou



# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

# Sets the name of the target
classes = ["Berggylte"]

# Saves all the image paths in an array
imagePaths = glob.glob(r"./dataset/testing/*.jpg")
imagePaths.sort()


layerNames = net.getLayerNames()
outputLayers = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(outputLayers)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Starts a counter that increments for each loop


# loop through all the images
for imagePath in imagePaths:

    # Loading image
    image = cv2.imread(imagePath)
    image = cv2.resize(image, None, fx=1, fy=1)
    height, width, channels = image.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    #
    net.setInput(blob)
    outs = net.forward(outputLayers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.001:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0,0,255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y + 30), font, 2, color, 1)


    image = cv2.rectangle(
        image,
        (int(dataList[index][1]), int(dataList[index][2])), 
        (int(dataList[index][3]), int(dataList[index][4])), 
        (0,255,0),
        thickness=2
    )

    ###
    # Uncomment this to show each image
    #cv2.imshow("Image", image)

    boxA = x, y, x+w, y+h
    boxB = int(dataList[index][1]), int(dataList[index][2]), int(dataList[index][3]), int(dataList[index][4])
    iouScore = bb_intersection_over_union(boxA, boxB)
    
    if iouScore < 0:
        iouScore = 0

    iouScores.append(iouScore)

    print("Image path: \t\t" + str(imagePath))
    print("Ground truth box: \t" + str(boxA))
    print("YoloV3 detected box: \t" + str(boxB))
    print("The iou score: \t\t" + str(iouScore))
    print("\n")
    
    
    cv2.imwrite("yoloResults/image00" + str(index) + ".jpg",image)
    print(index)
    
    index = index + 1

    ###
    # Uncomment this to go step by step
    #key = cv2.waitKey(0)
    ###

###
# Uncomment this to go step by step
#cv2.destroyAllWindows()
###

# Prints all the iou scores
print("All the iou scores: ")
print(iouScores)
print("\n")
print("The average iou score: ")
print(sum(iouScores)/50)

print("\n")
print("done")