import cv2
import os
from cube1 import myMain

def capture(name):
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame=cap.read()
        cv2.namedWindow(name)
        cv2.moveWindow(name,20,20)
        img=myMain(frame)
        cv2.imshow(name,img)
        key=cv2.waitKey(1)
        if key==98:
            path=os.path.dirname(os.path.realpath(__file__))
            path=path+"\\Images\\"
            if not os.path.exists(path):
                os.mkdir(path)
            cv2.imwrite(path+name+".png",frame)
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
    cv2.imshow(name,myMain(image))

capture("Left")
##capture("Right")
##capture("Front")
##capture("Back")
##capture("Top")
##capture("Bottom")

##load("cube")
##load("cube0")    
##load("cube1")    
##load("cube2")    
load("Left")    
##load("Bottom")    
    
cv2.waitKey(0)
cv2.destroyAllWindows()
