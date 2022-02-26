import cv2
import numpy as np
import threading
import time

from numpy.core.fromnumeric import put

# from matplotlib import pyplot as plt

lower = np.array([22,0,0])
upper = np.array([38,255,255])


def turn_left ():
    print ("turnleft")

def turn_Right ():
    print ("turnRight")
    
def direct ():
    print ("direct")

def stop ():
    print ("Stop")



def detact_navigation_object (img):
    return 

trafficconeTemplate1 = cv2.imread(".\\Cone\\1.png")


trafficconeTemplate = cv2.cvtColor(trafficconeTemplate1,cv2.COLOR_BGR2HSV)
trafficconeTemplate = cv2.inRange(trafficconeTemplate,lower,upper)

# cv2.imshow('trafficconeTemplate',trafficconeTemplate)

h, w = trafficconeTemplate.shape

SignTemplates = [trafficconeTemplate]
AllSigns = []
for scint in range(100,50,-50):
    scale = scint/100.0
    for stemp in SignTemplates:
        AllSigns.append(cv2.resize(stemp,(int(64*scale),int(64*scale))))


TemplateToString = {0:"Trafficcone"}
TemplateThreshold = 0.69

def processFrameConcurrent(idx, frame, template, rlist):
    res = cv2.matchTemplate(frame,template,cv2.TM_CCOEFF_NORMED)
    rlist.append((idx,cv2.minMaxLoc(res)))

def GetSignThread(imgframe):
    greyframe = cv2.cvtColor(imgframe, cv2.COLOR_BGR2GRAY)
    c = 0
    curMaxVal = 0
    curMaxTemplate = -1
    curMaxLoc = (0,0)
    ThreadList = []
    ReturnList = []
    for template in AllSigns:
        t = threading.Thread(target=processFrameConcurrent, args=(c,greyframe,template,ReturnList))
        t.daemon = True
        t.start()
        ThreadList.append(t)
        c = c + 1
    #Wait for each thread to finish
    for th in ThreadList:
        th.join()
    #process the returns
    for (idx, (min_val, max_val, min_loc, max_loc)) in ReturnList:
        if max_val > TemplateThreshold and max_val  > curMaxVal:
            curMaxVal = max_val
            curMaxTemplate = idx
            curMaxLoc = max_loc
        
    if curMaxTemplate == -1: 
        return (-1, (0,0),0, 0)
    else:
        return (curMaxTemplate%3, curMaxLoc, 1 - int(curMaxTemplate/3)*0.2, curMaxVal)

def GetSignSingle(imgframe):
    greyframe = cv2.cvtColor(imgframe, cv2.COLOR_BGR2GRAY)
    c = 0
    curMaxVal = 0
    curMaxTemplate = -1
    curMaxLoc = (0,0)


    for template in AllSigns:
        res = cv2.matchTemplate(greyframe,template,cv2.TM_CCOEFF_NORMED)  
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  #คือเมทตอดหรือฟังก์ชันตัวนึงที่จะรับ matchTemplate ให้คืนค่าออกมา ในที่คือ res 

        if max_val > TemplateThreshold and max_val  > curMaxVal:
            curMaxVal = max_val
            curMaxTemplate = c
            curMaxLoc = max_loc
        c = c + 1
    if curMaxTemplate == -1:
        return (-1, (0,0),0, 0)
    else:
        return (curMaxTemplate%3, curMaxLoc, 1 - int(curMaxTemplate/3)*0.2, curMaxVal)

        



cap = cv2.VideoCapture(0)
img = cv2.imread(".\\image\\safezone_val.jpeg")

img = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)


ret, frame = cap.read()

#บันทึกวิดีโอ
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output = cv2.VideoWriter('newvideo7.mp4',fourcc,20.0,(640,480)) 
    

hw,hh=(200,200)
focus=(800,500)
memory=10
center_memory=np.zeros((memory,2))
memory_i=0

while ret:

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower = np.array([22,0,0])
    upper = np.array([38,255,255])

    detest = cv2.inRange(hsv,lower,upper)

    copyimg = img.copy()

    if not ret:
        break
    template=-1
    
    (template, top_left, scale, val) = GetSignSingle(frame)

    if template != -1:
       
        res = cv2.matchTemplate(detest, trafficconeTemplate, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[-1] + h)

        center_match=(max_loc[1]+int(h/2),max_loc[0]+int(w/2))
        if center_match is not None or len(center_match)==2:
            center_memory[memory_i]=center_match
            center_match=np.median(center_memory,axis=0).astype(np.int16)
            memory_i=(memory_i+1)%memory
            
            
            #Safe zone บนภาพวาด
            safezone_val = img[center_match[0], center_match[1]]
            if safezone_val > 200 : 
                cv2.putText(frame,'turn_left',(20,450),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(23,55,0),2)
                turn_left()
            elif safezone_val < 100 :
                cv2.putText(frame,'turn_Right',(20,450),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(23,55,0),2)
                turn_Right()
            else :
                cv2.putText(frame,'direct',(20,450),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(23,55,0),2)
                print("direct")



            print(center_match)
            print(img[center_match[0], center_match[1]])

            #Safe zoon บนจอภาพ
            safezone_val = img[center_match[0], center_match[1]]
            if safezone_val > 200 : 
                cv2.putText(copyimg,'turn_left',(20,300),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,0),2)
                turn_left()
            elif safezone_val < 100 :
                cv2.putText(copyimg,'turn_Right',(20,300),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,0),2)
                turn_Right()
            else :
                cv2.putText(copyimg,'direct',(20,300),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,0),2)
                print("ตรงไป")


        frame[center_match[0]:center_match[0]+10,center_match[1]:center_match[1]+10]=255

        

        copyimg[center_match[0]:center_match[0]+10,center_match[1]:center_match[1]+10]=100

        
        # cv2.putText(frame,str(val),(20,300),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
        # cv2.putText(frame,str(center_match),(20,300),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,0),2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 3)
        cv2.putText(frame,TemplateToString[template],(20,400),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)


    #บันทึกวิดีโอ
    # if(ret==True):
    #     output.write(frame)
    
    
    #print(center_match)
    # print(res.shape)
    # print(frame.shape)
    # print(trafficconeTemplate.shape)
    # print([min_val, max_val, min_loc, max_loc])
    # detest = cv2.cvtColor(detest,cv2.COLOR_)

    # cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 3)

    #cv2.imshow("Crop",frame[270:480,200:420,:])
    #cv2.imshow("Result",res)
    cv2.imshow("Original",frame)
    cv2.imshow("Img",copyimg)
    # cv2.imshow('Frame2',frame[focus[1]-hh:focus[1]+hh,focus[0]-hw:focus[0]+hw,:])
    # cv2.imshow('Detest Result',detest)
    
    ret, frame = cap.read()
    
    if cv2.waitKey(1) & 0xFF== ord('q'):
                break

    #บันทึกวิดีโอ
# cap.release()

cv2.destroyAllWindows



