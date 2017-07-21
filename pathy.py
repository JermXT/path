#cat img > input.txt | python pathy.py

import numpy as np
import cv2, sys, math, codecs, time




def path(img, k): 
        object_probability = 0

#0 100 200        
#170 255 255
        blue = 0
        lower = np.array([0,100,200])
        upper = np.array([185, 230, 255])
        

        mask = cv2.inRange(img, lower, upper)
        while cv2.countNonZero(mask) < 700 and blue < 245:
            lower_high =  np.array([0,100, 200])
            upper_high = np.array([blue,255, 255])
            mask = cv2.inRange(img, lower_high, upper_high)

            if blue > 190:
                object_probability = object_probability+5
                blue +=5
            else:
                blue += 10
#finds pixels that are more orange than blue
#if less than 700 pixels are found, then the requirements are loosened

        n = 1
        if blue > 210: 
            n = int(object_probability/15)
     


        kernel = np.ones((2,1),np.uint8)
        #kernel = np.ones((1, 1),np.uint8)
        dilation = cv2.dilate(mask,kernel,iterations = n)
#dialates pixels to make finding large contours easier
#dialation amount dependent on how much orange there is in image



        areaArray = [] 
        largestArea = -1
        largestIndex = -1

        ret, thresh = cv2.threshold(dilation,127,255,0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > largestArea:
                largestArea = area
                largestIndex = i

        if largestIndex < 0:
            return
#finds largest contour

        white_pixels = cv2.countNonZero(dilation)

        dilation = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)
#draws contour
        


        cnt = contours[largestIndex]
#        cv2.drawContours(dilation, contours, count, (180,180, 180), 2)   
#        cv2.rectangle(dilation, (x, y), (x+w, y+h), (180,180,180), 2) 
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(dilation,[box],0,(0,255,0),2)
#draws box around largest contour



        l1 = (box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2 
        l2 = (box[1][0] - box[2][0]) ** 2 + (box[1][1] - box[2][1]) ** 2
         
        m = 0

#inverted y values because down is +y
        if (l1 > l2):
            if box[0][0]-box[1][0] == 0:
                m = 'undef' 
            else:
                m = -(box[0][1] - box[1][1]) / (box[0][0] - box[1][0])
        else:
            if box[1][0]-box[2][0] == 0:
                m = 'undef' 
            else:
                m = -(box[1][1] - box[2][1]) / (box[1][0] - box[2][0])
        if m != 'undef':
            theta = math.degrees(math.atan(m))
        else:
            theta = 90
        if (theta < 0):
            theta += 180
        bearing = 90 - theta
        
#        print(bearing)
#finds angle of path
        
        if l1 > l2:
            r = math.sqrt(l1)
        else:
            r = math.sqrt(l2)


        x_center = int((box[0][0]+box[1][0]+box[2][0]+box[3][0])/4)
        y_center = int((box[0][1] + box[1][1] + box[2][1] + box[3][1])/4)

        cv2.circle(dilation, (int(x_center), int(y_center)), 2, (255,125,125), thickness = 6, lineType = 8)
#finds center of path



        length = 100
        x2 =  int(x_center + length * math.cos(theta * math.pi / 180.0))
        y2 =  int(y_center - length * math.sin(theta * math.pi / 180.0))
       
        cv2.line(dilation,(x_center, y_center), (x2, y2), (100, 100, 255), thickness = 4)
#draws line in direction of path
         
        contour_area = cv2.contourArea(contours[largestIndex])

        
        fill = cv2.contourArea(contours[largestIndex])/(math.sqrt(l1)*math.sqrt(l2))
        #% of rectangle is of path pixels
        
        
        sides = [math.sqrt(l1), math.sqrt(l2)]
        ratio = max(sides)/min(sides) 
        #ratio of long side to short side

        noise = float(white_pixels/contour_area)
        #if noise is large, lots of orange particles not in "path"
        
        confidence = 0

        if noise < 3:
            confidence = confidence + 0.7
        elif noise < 4:
            confidence = confidence + 0.3
        if ratio >10 and noise > 3:
            confidence = confidence + 0.5
        confidence = confidence + fill
        if confidence > 1:
            confidence = 1

#probability that path exists        
        return ((r,theta), (x_center, y_center), confidence, dilation)
        #cv2.imwrite('../vision/logs/path_log_'+time.strftime('%x_%X')+'.png', img)
        # writes log picture     




def vision_func(slope, position, confidence):
    img = cv2.imread("input.txt")
    slope, position, confidence, dilation = path(img, 255)
    return slope, position, confidence, dilation

slope, position, confidence, dilation = vision_func(0,0,0)






#testing code
'''
for i in range(10, 70):
        #print(i)
        img = cv2.imread("path_training_" + str(i) + ".png")
        cv2.imshow("image", img)
##        path(img, 32)
        a,b,c,dilation = path(img, 255)
        print (a,b,c)
        print(" ")
        cv2.imshow("default", dilation)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
'''

