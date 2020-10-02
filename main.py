import cv2
import numpy as np

class Koordinat:
    def __init__(self,x,y):
        self.x = x
        self.y = y
class Sensör():
    def __init__(self,Koordinat1,Koordinat2,w_cropped,h_cropped,serit):
        self.Koordinat1 = Koordinat1
        self.Koordinat2 = Koordinat2
        self.w = abs(Koordinat1.x-Koordinat2.x)
        self.h = abs(Koordinat1.y-Koordinat2.y)
        self.area = abs((Koordinat2.x-Koordinat1.x)*(Koordinat2.y-Koordinat1.y))
        self.status = False
        self.Maske = np.zeros((w_cropped,h_cropped,1),np.uint8)
        self.counter = 0
        self.serit=serit
        cv2.rectangle(self.Maske,(Koordinat1.x,Koordinat1.y),(Koordinat2.x,Koordinat2.y),(255,255,255),-1)

    def changeStatus(self,status):
        self.status=status

video_r = cv2.VideoCapture("Relaxing highway traffic.mp4")
backSub = cv2.createBackgroundSubtractorMOG2()
kernel1=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

Sensör2 = Sensör(Koordinat(350,210),Koordinat(470,300),320,1280,2)
Sensör1 = Sensör(Koordinat(120,210),Koordinat(240,300),320,1280,1)
Sensör3 = Sensör(Koordinat(830,210),Koordinat(910,270),320,1280,3)
Sensör4 = Sensör(Koordinat(980,210),Koordinat(1090,300),320,1280,4)
Sensör_list = [Sensör1,Sensör2,Sensör3,Sensör4]
sum_counter = 0

while(1):
    ret,kare = video_r.read()

    cropped_image = kare[400:,:]
    filtered_image = backSub.apply(cropped_image)
    filtered_image = cv2.morphologyEx(filtered_image,cv2.MORPH_OPEN,kernel1)

    ret,filtered_image = cv2.threshold(filtered_image,127,255,cv2.THRESH_BINARY)
    copied_image = cropped_image.copy()

    contoured_img = np.zeros((filtered_image.shape[0],filtered_image.shape[1],1),np.uint8)

    _,thresh = cv2.threshold(filtered_image,80,255,0)
    contours,__ = cv2.findContours(thresh,1,2)
    for cnt in contours:
        if cv2.contourArea(cnt)>1500:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(copied_image,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.rectangle(contoured_img,(x,y),(x+w,y+h),255,-1)

    for i in Sensör_list:
        cv2.rectangle(copied_image, (i.Koordinat1.x, i.Koordinat1.y),
                      (i.Koordinat2.x, i.Koordinat2.y), (0, 255, 0), -1)

    for i in Sensör_list:
        mask_combined = cv2.bitwise_and(contoured_img, contoured_img, mask=i.Maske)
        rate=np.sum(mask_combined)/i.area

        if rate>200:
            if i.status==False:
                cv2.rectangle(copied_image, (i.Koordinat1.x, i.Koordinat1.y),
                              (i.Koordinat2.x, i.Koordinat2.y), (0, 0, 255), -1)
                i.changeStatus(True)
                i.counter+=1
                sum_counter+=1
                print("Toplam gecen arac sayisi: "+str(sum_counter))
            else:
                cv2.rectangle(copied_image, (i.Koordinat1.x, i.Koordinat1.y),
                              (i.Koordinat2.x, i.Koordinat2.y), (0, 0, 255), -1)
                i.changeStatus(True)
        else:
            cv2.rectangle(copied_image, (i.Koordinat1.x, i.Koordinat1.y),
                          (i.Koordinat2.x, i.Koordinat2.y), (0, 255, 0), -1)
            i.changeStatus(False)

    for i in Sensör_list:
        cv2.putText(copied_image, str(i.counter), (i.Koordinat1.x, i.Koordinat1.y + 60), cv2.FONT_ITALIC, 2, (255, 255, 255), 4)

    cv2.imshow("mask_combined",mask_combined)
    cv2.imshow("cropped image",copied_image)

    k=cv2.waitKey(30) & 0xff

    if k==27:
        break

video_r.release()
cv2.destroyAllWindows()

