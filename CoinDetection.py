import cv2 as cv
import numpy as np
cap=cv.VideoCapture(2)

while(cap.read()) :
     ref,frame = cap.read()

     gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
     gray_blur=cv.GaussianBlur(gray,(15,15),0)
     thresh=cv.adaptiveThreshold(gray_blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,1)
     kernel=np.ones((1,1),np.uint8)
     closing=cv.morphologyEx(thresh,cv.MORPH_CLOSE,kernel,iterations=3)

     result_img=closing.copy()
     contours,hierachy=cv.findContours(result_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
     for cnt in contours:
         area = cv.contourArea(cnt)
         if area>500  :
             coords = cv.boundingRect(cnt) 
             width = (coords[0]+(coords[0]+coords[2]))/2
             height = (coords[1]+(coords[1]+coords[3]))/2
             centerPoint = (int(width),int(height))
             cv.rectangle(frame,(coords[0],coords[1]),(coords[0]+coords[2],coords[1]+coords[3]),(0,0,255),2)
             cv.circle(frame,centerPoint,2,(0,0,255),1)
             print(centerPoint)
     cv.imshow("Show",frame)
     if cv.waitKey(1) & 0xFF==ord('q'):
         break

cap.release()
cv.destroyAllWindows()
