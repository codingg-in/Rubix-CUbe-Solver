import cv2
import os
from cube1 import myMain

def capture(name):
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame=cap.read()
        cv2.namedWindow(name)
        cv2.moveWindow(name,20,20)
        rotate, img = myMain(frame, 1)
        cv2.imshow(name, img)
        key=cv2.waitKey(1)
        if key==98:
            path=os.path.dirname(os.path.realpath(__file__))
            path=path+"\\Images\\"
            if not os.path.exists(path):
                os.mkdir(path)
            cv2.imwrite(path+name+".png",frame)
            cv2.imwrite(path+name+"1.png",frame)

            break
    
    cap.release()
    cv2.destroyAllWindows()


def load(name):
    path=os.path.dirname(os.path.realpath(__file__))
    path=path+"\\Images\\"+name+".png"
    cv2.namedWindow(name)
    cv2.moveWindow(name,20,20)
    image=cv2.imread(path)
    #print (image)
    rotate, image = myMain(image, 0)
    if rotate == 0:
        cv2.imshow(name, image)
        cv2.waitKey(0)
    return rotate

capture("Left")
##capture("Right")
##capture("Front")
##capture("Back")
##capture("Top")
##capture("Bottom")

rotate = load("Left")
if rotate == 1:
    rotate = load("Left")
##load("Right")    
##load("Front")    
##load("Back")    
##load("Top")    
##load("Bottom")    
    
cv2.waitKey(0)
cv2.destroyAllWindows()
