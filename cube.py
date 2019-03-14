import cv2
import os

def capture(name):
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame=cap.read()
        cv2.namedWindow(name)
        cv2.moveWindow(name,20,20)
        cv2.imshow(name,frame)
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
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#capture("Left")
#capture("Right")
#capture("Front")
#capture("Back")
#capture("Top")
#capture("Bottom")

load("Left")    
load("Right")    
load("Front")    
load("Back")    
load("Top")    
load("Bottom")    
    
