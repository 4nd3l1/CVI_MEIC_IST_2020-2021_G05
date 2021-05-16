from typing import Counter
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET


def loadVideo(folder, name):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)

    height, width, layers = images[0].shape
    size = (width,height)        
    global video 
    video = cv2.VideoWriter(os.path.join("Assignment2",name) + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 7, size)
    for i in range(len(images)):
        video.write(images[i])
    video.release()  

def buildGT():
    frame = 0
    tree = ET.parse(os.path.join("Assignment2","PETS2009-S2L1.xml"))
    root = tree.getroot()
    for filename in os.listdir("Assignment2/S2_L1"):
        img = cv2.imread(os.path.join("Assignment2/S2_L1",filename))
        for child in root[frame][0]:
            h = int(float(child[0].attrib['h']))
            w = int(float(child[0].attrib['w']))
            xc = int(float(child[0].attrib['xc']))
            yc = int(float( child[0].attrib['yc']))
            img = cv2.rectangle(img, (xc - int(w / 2), yc - int(h / 2)), (xc - int(w / 2) + w, yc - int(h / 2) + h), (255, 0, 0), 2)
        cv2.imwrite(os.path.join("Assignment2/S2_L1_GT",filename), img)
        frame = frame + 1

def pedDetect():
    
    video = cv2.VideoCapture(os.path.join("Assignment2","project.avi"))
    ret, frame1 = video.read()
    ret, frame2 = video.read()

    track_size = 20
    track_points = []

    ind = 0

    while video.isOpened():
        diff = cv2.absdiff(frame1, frame2);
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 900:
                continue
            frame1 = cv2.rectangle(frame1, (x,y), (x+w,y+h), (255,0,0), 2)
            name = f"{ind:04}"
            cv2.imwrite(os.path.join("Assignment2/S2_L1_A","frame_" + name + ".jpg"), frame1)

            # Centroid
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            track_points.append((cX, cY, track_size))
        
        for i in range(0,len(track_points)-1):
                if(track_points[i][2] != 0):
                    (x,y,c) = track_points[i]
                    cv2.circle(frame1, (x, y), 2, (0, 0, 255), -1)
                    l = list(track_points[i])
                    l[2] = l[2] - 1
                    track_points[i] = tuple(l)

        cv2.imshow("feed", frame1)
        frame1 = frame2
        ret, frame2 = video.read()
        if ret:
            ind = ind + 1
            if cv2.waitKey(40) == 27:
                break;
        else:
            break

    cv2.destroyAllWindows()
    video.release()

buildGT()
loadVideo("Assignment2/S2_L1", "project")
loadVideo("Assignment2/S2_L1_GT", "reference")
pedDetect()
loadVideo("Assignment2/S2_L1_A", "algorithm")
