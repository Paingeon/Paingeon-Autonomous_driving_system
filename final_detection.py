from array import array
from re import A
import cv2
from cv2 import imshow
import numpy as np
import threading
import time
from numpy.core.fromnumeric import put
# from matplotlib import pyplot as plt

#รายชื่อหมวดหมู่ทั้งหมด เรียงตามลำดับ
CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
	"BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
	"DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
	"SOFA", "TRAIN", "TVMONITOR"]
#สีตัวกรอบที่วาดrandomใหม่ทุกครั้ง
COLORS = np.random.uniform(0,100, size=(len(CLASSES), 3))
#โหลดmodelจากแฟ้ม
net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD.prototxt","./MobileNetSSD/MobileNetSSD.caffemodel")


lower = np.array([22,0,0])
upper = np.array([38,255,255])


def turn_left ():
    print ("turnleft")

def turn_right ():
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

        



cap = cv2.VideoCapture(1)
img = cv2.imread(".\\image\\safezone_val3.png")
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

    # if not ret:
    #     break
    if ret:
        (h,w) = frame.shape[:2]
        #ทำpreprocessing
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
        net.setInput(blob)
        #feedเข้าmodelพร้อมได้ผลลัพธ์ทั้งหมดเก็บมาในตัวแปร detections
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            percent = detections[0,0,i,2]
            #กรองเอาเฉพาะค่าpercentที่สูงกว่า0.5 
            if percent > 0.5:
                class_index = int(detections[0,0,i,1])
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype("int")
                # box.astype("int") == A

                aruyo = (frame, (startX, startY), (endX, endY), COLORS[class_index], 2)
                # aruyo = frame[A] 
                # cv2.imshow(frame[box.astype("int")])
                # cv2.imshow(frame(box.astype("int")))
                # aruyo = frame[[startX, startY], [endX, endY]]
                # print(startX, startY, endX, endY)
                                
            #ส่วนตกแต่ง วาดกรอบและชื่อ
                label = "{} [{:.2f}%]".format(CLASSES[class_index], percent*100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_index], 2)
                cv2.rectangle(frame, (startX-1, startY-30), (endX+1, startY), COLORS[class_index], cv2.FILLED)
                cv2.putText(frame,'stop',(20,300),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255),2)
                y = startY - 15 if startY-15>15 else startY+15
                cv2.putText(frame, label, (startX+20, y+5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
                
            cv2.imshow("person",aruyo)
                
        
        

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
            
            
            #Safe zone บนจอวาด
            safezone_val = img[center_match[0], center_match[1]]
            if safezone_val > 200 : 
                cv2.putText(frame,'Turn_left',(20,450),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(23,55,255),2)
                turn_left()
            elif safezone_val < 100 :
                cv2.putText(frame,'Turn_Right',(20,450),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(23,55,255),2)
                turn_right()
            # elif percent > 0.5 :
            #     cv2.putText(copyimg,'stop',(20,300),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
            #     stop()
            else :
                cv2.putText(frame,'Direct',(20,450),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(23,55,255),2)
                print("direct")
                
        
            print(center_match)
            print(img[center_match[0], center_match[1]])

            #Safe zoon บนภาพวาด
            safezone_val = img[center_match[0], center_match[1]]
            if safezone_val > 200 : 
                cv2.putText(copyimg,'Turn_left',(20,300),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
                turn_left()
            elif safezone_val < 100 :
                cv2.putText(copyimg,'Turn_Right',(20,300),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
                turn_right()
            # elif percent > 0.5 :
            #     cv2.putText(copyimg,'stop',(20,300),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
            #     stop()
            else :
                cv2.putText(copyimg,'Direct',(20,300),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
                print("direct")
        
        if ret :
            cv2.putText(copyimg,'stop',(20,500),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
            stop()



    
        frame[center_match[0]:center_match[0]+10,center_match[1]:center_match[1]+10]=255
        copyimg[center_match[0]:center_match[0]+10,center_match[1]:center_match[1]+10]=100

        
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 3)
        cv2.putText(frame,TemplateToString[template],(20,400),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)


    #บันทึกวิดีโอ
    # if(ret==True):
    #     output.write(frame)
    
    cv2.imshow("Original",frame)
    cv2.imshow("Img",copyimg)
    
    

   
    
    ret, frame = cap.read()
    
    if cv2.waitKey(1) & 0xFF== ord('q'):
                break

    #บันทึกวิดีโอ
cap.release()
cv2.destroyAllWindows