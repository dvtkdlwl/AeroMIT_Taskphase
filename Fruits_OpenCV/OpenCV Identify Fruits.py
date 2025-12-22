import cv2
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread("Farm.jpeg")
img2=img.copy()

hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#Finding purple mask:
lower_purple=np.array([125,170,170])
upper_purple=np.array([255,255,255])

maskpurple=cv2.inRange(hsv,lower_purple,upper_purple)
respurple=cv2.bitwise_and(img,img,mask=maskpurple)

#Finding red mask:
lower_red1=np.array([0,120,70])
upper_red1=np.array([10,255,255])

lower_red2=np.array([170,120,70])
upper_red2=np.array([180,255,255])

#Here it is combining both the masks which target different red shades:
maskred1=cv2.inRange(hsv,lower_red1,upper_red1)
maskred2=cv2.inRange(hsv,lower_red2,upper_red2)
maskred=cv2.bitwise_or(maskred1,maskred2)
resred=cv2.bitwise_and(img2,img2,mask=maskred)

kernel=np.ones((6,6), np.uint8)
dilation=cv2.dilate(maskred,kernel,iterations=1)

#Copying the image for finding contours:
imgcontour=img.copy()

#Showing all the images obtained:
#cv2.imshow("mask", maskred)
#cv2.imshow("dilated", dilation)
#cv2.imshow("res", resred)
cv2.imshow("maskpurple", maskpurple)
#cv2.imshow("res", respurple)


#Drawing bounding rectangles for all the fruits while using an area filter:
def getcontours(img):
    contours, hierarchy= cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if 20< cv2.contourArea(cnt)<500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(imgcontour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if np.any(maskred[y:y+h, x:x+w]):
                cv2.putText(imgcontour, "Red Fruit", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
            elif np.any(maskpurple[y:y+h, x:x+w]):
                cv2.putText(imgcontour, "Purple Fruit", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
            

        

cv2.imshow("img", img)
getcontours(maskpurple)
getcontours(dilation)
cv2.imshow("imgcontour", imgcontour)


cv2.waitKey(0)
cv2.destroyAllWindows()