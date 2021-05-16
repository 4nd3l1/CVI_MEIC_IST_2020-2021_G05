import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

import random

import math 

path = "Assignment1\MATERIAL\database\cvi_1.png"

img_original = cv2.imread(path, 1)
img_selected = []
selected = list()
numberOfParts = 1

cnts = []

def get_cnts(image):
    image_blur = cv2.medianBlur(image,21)
    image_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    thresh, image_thresh1 = cv2.threshold(image_gray,230,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    image_open = cv2.morphologyEx(image_thresh1,cv2.MORPH_OPEN,kernel,iterations = 2) 
    image_dilated = cv2.dilate(image_open,kernel,iterations=3)
    image_distance = cv2.distanceTransform(image_dilated,cv2.DIST_L2,5)
    thresh1, image_thresh2 =  cv2.threshold(image_distance, 0.1*image_distance.max(),255,0)
    image_thresh2_8 = np.uint8(image_thresh2)
    cnts = cv2.findContours(image_thresh2_8.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def count_centroid():
    image = img_original.copy()
    for (i, c) in enumerate(cnts):
        ((x, y), _) = cv2.minEnclosingCircle(c)
        cv2.putText(image, "#{}".format(i + 1), (int(x) - 45, int(y)+20),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        cv2.drawContours(image, [c], 0, (0, 255, 0), 2)
        #centroid
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(image, (cX, cY), 10, (0, 255, 0), -1)
    count = len(cnts)
    return image,count

def hconcat_resize(img_list, interpolation = cv2.INTER_CUBIC):
      # take minimum hights
    h_min = max(img.shape[0] 
                        for img in img_list)
                        
      
    # image resizing 
    im_list_resize = [cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min), interpolation = interpolation) 
                        for img in img_list]
      
    # return final image
    return cv2.hconcat(im_list_resize)

def get_objects():
    image_copy = img_original.copy()
    splitted = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        splitted.append(image_copy[y:y+h, x:x+w])
    return splitted

def area():
    image_copy = img_original.copy()
    cntsSorted = sorted(cnts, key=cv2.contourArea, reverse = True)
    area_values = [cv2.contourArea(cnt) for cnt in cntsSorted]
    splitted = get_objects()
    for i in range(0, len(splitted)):
        img = splitted[i]
        area = area_values[i]
        height, width, _ = img.shape
        label_width, label_height = cv2.getTextSize("{:.2f}".format(area), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        cv2.putText(img, "{:.2f}".format(area), (0,  height - label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    final_img = hconcat_resize(splitted)
    return final_img

def getBlurValue(image):
    canny = cv2.Canny(image,50,250)
    return np.mean(canny)
   
def blurriness():
    image_copy = img_original.copy()
    splitted = get_objects()
    ordered = [(img, getBlurValue(img)) for img in splitted]
    ordered = sorted(ordered, key=lambda tup: tup[1])
    for (img, blur) in ordered:
        height, width, _ = img.shape
        label_width, label_height = cv2.getTextSize("{:.2f}".format(blur), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, "{:.2f}".format(blur), (0,  height - label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    ordered_single = [img for (img,blur) in ordered]
    final_img = hconcat_resize(ordered_single)
    return final_img

def rotate(image, angle):
    image_copy = image.copy()
    image_center = tuple(np.array(image_copy.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image_copy, rot_mat, image_copy.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def get_similarity(image_a, image_b):
    obj_gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    histogram_a = cv2.calcHist([obj_gray_a], [0], None, [256], [0, 256])
    obj_gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    histogram_b = cv2.calcHist([obj_gray_b], [0], None, [256], [0, 256])
    c1 = 0
    i = 0
    while i<len(histogram_a) and i<len(obj_gray_b):
        c1+=(histogram_a[i]-histogram_b[i])**2
        i+= 1
    c1 = c1**(1 / 2)
    return c1

def hough_circle_detection(coins):
    # turn original image to grayscale
    gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)  
    # blur grayscale image
    blurred = cv2.medianBlur(gray, 5)  
    return cv2.HoughCircles(
        blurred,  # source image (blurred and grayscaled)
        cv2.HOUGH_GRADIENT,  # type of detection
        1,  # inverse ratio of accumulator res. to image res.
        50,  # minimum distance between the centers of circles
        param1=50,  # Gradient value passed to edge detection
        param2=50,  # accumulator threshold for the circle centers
        minRadius=10,  # min circle radius
        maxRadius=380,  # max circle radius
    )  

def countMoney():
    euros = {
        "1 Cent": {
            "value": 0.01,
            "radius": 8.13,
            "ratio": 1,
            "count": 0,
        },
        "2 Cent": {
            "value": 0.02,
            "radius": 9.38,
            "ratio": 1.154,
            "count": 0,
        },
        "5 Cent": {
            "value": 0.05,
            "radius": 10.63,
            "ratio": 1.308,
            "count": 0,
        },
        "10 Cent": {
            "value": 0.10,
            "radius": 9.88,
            "ratio": 1.215,
            "count": 0,
        },
        "20 Cent": {
            "value": 0.20,
            "radius": 1.370,
            "ratio": 1.3,
            "count": 0,
        },
        "50 Cent": {
            "value": 0.50,
            "radius": 12.1,
            "ratio": 1.492,
            "count": 0,
        },
        "1 Euro": {
            "value": 1.0,
            "radius": 11.5,
            "ratio": 1.431,
            "count": 0,
        },
        "2 Euro": {
            "value": 2.0,
            "radius": 12.5,
            "ratio": 1.585,
            "count": 0,
        },
    }

    circles = hough_circle_detection(img_original.copy())
    radius = []
    coordinates = []

    cv2.namedWindow("Object Selection")

    for detected_circle in circles[0]:
        x_coor, y_coor, detected_radius = detected_circle
        radius.append(detected_radius)
        coordinates.append([x_coor, y_coor])

    smallest = min(radius)
    tolerance = 0.03
    total_amount = 0
    print(smallest)

    for coin in circles[0]:
        ratio_to_check = coin[2] / smallest
        coor_x = coin[0]
        coor_y = coin[1]
        for euro in euros:
            value = euros[euro]['value']
            if abs(ratio_to_check - euros[euro]['ratio']) <= tolerance:
                euros[euro]['count'] += 1
                total_amount += euros[euro]['value']
    return total_amount

def heatmap(image):
    hm = image.copy()
    cnts_hm = get_cnts(hm)
    M = cv2.moments(cnts_hm[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    height, width, _ = hm.shape
    max_dist = math.sqrt((width/2)**2 + height/2**2)
    for x in range(0, width):
        for y in range(0, height):
            dist = cv2.pointPolygonTest(cnts_hm[0], (y,x), True)
            col = (dist * 255) / max_dist
            hm[y,x] = (255 - col, 0, col)
    return hm

def display(cmap="gray"):
    img,count_n = count_centroid();
    img_area_sorted = area()
    img_sharpness_sorted = blurriness()
    angle = 90;
    objects = get_objects()
    img_1_rotated = rotate(objects[random.randint(0, len(objects)-1)], angle)

    global f, axs
    f, axs = plt.subplots(3,2,figsize=(12,5))
    f.tight_layout(pad=3.0)
    axs[0, 0].imshow(img_original[:,:,::-1])
    axs[0, 0].set_title("Original Image")
    axs[0, 0].margins(3, 3)
    axs[0, 1].imshow(img[:,:,::-1])
    axs[0, 1].set_title("Total Count = {}".format(count_n))
    axs[1, 0].imshow(img_area_sorted[:,:,::-1])
    axs[1, 0].set_title("Sorted by Area")
    axs[1, 1].imshow(img_sharpness_sorted[:,:,::-1])
    axs[1, 1].set_title("Sorted by Blurriness")
    axs[2, 0].imshow(img_1_rotated[:,:,::-1])
    axs[2, 0].set_title("Random object rotated {}º clockwise".format(angle))
    axs[2, 1].imshow(img_original[:,:,::-1])
    axs[2, 1].set_title("Amount of Money: {:.2f}".format(countMoney()) )
   
    plt.show()
    cv2.waitKey(0);
    cv2.destroyAllWindows();

def displaySelected(cmap="gray"):
    global numberOfParts
    numberOfParts = 1
    img = img_selected.copy()
    img_cnts = get_cnts(img)

    #centroid
    ((x, y), _) = cv2.minEnclosingCircle(img_cnts[0])
    cv2.drawContours(img, [img_cnts[0]], 0, (0, 255, 0), 1)
    M = cv2.moments(img_cnts[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(img, (cX, cY), 2, (0, 255, 0), -1)
    selected.append(img)

    #area
    area = cv2.contourArea(img_cnts[0])
    #blur
    blur = getBlurValue(img)
    #transform - rotated
    angle = 45
    rot = rotate(img, angle)
    numberOfParts = numberOfParts + 1
    
    
    img = hconcat_resize([img, rot])
    #organized by similarity
    objs = get_objects()
    for obj in objs:
        if obj.shape == img_selected.shape and not(np.bitwise_xor(obj,img_selected).any()):
            objs.remove(obj)
    ordered = [(i, get_similarity(img_selected, i)) for i in objs]
    ordered = sorted(ordered, key=lambda tup: tup[1], reverse=False)
    numberOfParts+=len(ordered)
    ordered_single = [i for (i,s) in ordered]
    sim_concat = hconcat_resize(ordered_single)
    img = hconcat_resize([img, sim_concat])
    #heatmap
    img = hconcat_resize([img, heatmap(img_selected)])
    numberOfParts = numberOfParts + 1    
    height, width, _ = img.shape
    scaled = cv2.resize(img, (width*2, height *2)) 
    height, width, _ = scaled.shape
    #add_text_tl(scaled, "Selected:")
    label_width, label_height = cv2.getTextSize("Area: " + "{:.2f}".format(area), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(scaled, "Area: " + "{:.2f}".format(area), (0,  label_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    label_width, label_height = cv2.getTextSize("Blur: " + "{:.2f}".format(blur), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(scaled, "Blur: " + "{:.2f}".format(blur), (0,  label_height * 3 * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    label_width, label_height = cv2.getTextSize("Rotated by: {} deg".format(angle), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(scaled, "Rotated by: {} deg".format(angle), (round(width/numberOfParts),  label_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    index = 0
    for (i, s) in ordered:
        label_width, label_height = cv2.getTextSize("Sim: " + "{:.2f}".format(s[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(scaled, "Diff: " + "{:.2f}".format(s[0]), ((round(width/numberOfParts)) + (round(width/numberOfParts) * (index + 1)),  label_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        index = index + 1
    
    cv2.namedWindow("Selected" + str(len(selected)))
    cv2.imshow("Selected" + str(len(selected)), scaled)
    cv2.waitKey(0);
    cv2.destroyAllWindows();

def click_and_crop(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        print(refPt)
        for c in cnts:
            ptx,pty,w,h = cv2.boundingRect(c)
            if(x >= ptx and y >= pty and x <= ptx+w and y <= pty+h):
                print("Encontrei um objeto")
                global img_selected
                img_selected = img_original[pty:pty+h, ptx:ptx+w]
                displaySelected()

cv2.namedWindow("Object Selection")
cv2.setMouseCallback("Object Selection", click_and_crop)
cv2.imshow("Object Selection", img_original)

cnts = get_cnts(img_original)


display()


