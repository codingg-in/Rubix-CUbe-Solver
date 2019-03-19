import cv2
import numpy as np

image=cv2.imread("cube.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
canny = cv2.Canny(blurred, 20, 40)
#cv2.imshow("try",canny)

kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(canny, kernel, iterations=3)
#cv2.imshow("3",dilated)

(contours, hierarchy) = cv2.findContours(dilated.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image, contours, -1, (255,0,0), 2)
#cv2.imshow("1",image)
#print(contours[0])
square_contours=[]
square_in_square=[]
for cnt in contours:
    #cnt=contours[i]
    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    if len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w/float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        if ar >= 0.90 and ar <= 1.1:
            square_contours.append(approx)

##for sc in square_contours:
##    cv2.drawContours(image, [sc], -1, (0,255,0), 2)
##
##print(square_contours)
##cv2.drawContours(image, [square_contours[30]], -1, (0,255,0), 2)
##cv2.drawContours(image, [square_contours[31]], -1, (0,255,0), 2)
#for sc in square_contours:
##for i in range(40):
##    cv2.drawContours(image, [square_contours[i]], -1, (0,255,0), 2)
##    cv2.imshow("5",image)
##    cv2.waitKey(0)
  

##print(square_contours[30])
##print(square_contours[31])

for j in range(0,len(square_contours)):
    check=0
##    print('\n')
    (x, y, w, h) = cv2.boundingRect(square_contours[j])
    for i in range(j+1,len(square_contours)):
        (xx, yy, ww, hh) = cv2.boundingRect(square_contours[i])
        if (x<=xx) and ((x+w)>=(xx+ww)) and (y<yy) and ((y+h)>=(yy+hh)):
            print(x,y,w,h)
            square_in_square.append(square_contours[i])

##check=1
##    if check==0:

sc_sis=[]
##for i in range(0,40):
##    if(square_contours[i][1]


##print(square_contours[31])
##print(dsc[3][1])
##print(dsc[4])
##dsc[3][1][0]
print (len(square_contours))
print (len(square_in_square))
print(len(sc_sis))
k=0
if(len(square_in_square)>0):
    for j in range(0,len(square_contours)):
        if (square_contours[j]!=square_in_square[k]).all():
            sc_sis.append(square_contours[j])
        else:
            print(j,"+",k)
            k=k+1
        if k==len(square_in_square):
            k=k-1
elif len(square_in_square)==0:
    for j in range(0,len(square_contours)):
            sc_sis.append(square_contours[j])
    
        
        


for sc in sc_sis:
    cv2.drawContours(image, [sc], -1, (0,255,0), 2)
#print(xx,yy,ww,hh)
##print(square_contours[32])
##print(square_contours[33])


cv2.imshow("5",image)

cv2.waitKey(0)
cv2.destroyAllWindows()
