import cv2
import numpy as np
import enum

class Koordinat:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Sensör:
    def __init__(self, Koordinat1, Koordinat2, Uzunluk, Genislik):
        self.Koordinat1 = Koordinat1
        self.Koordinat2 = Koordinat2
        self.Genislik = Genislik
        self.Status = False
        self.Uzunluk = Uzunluk
        self.Maske = np.zeros((Uzunluk,Genislik,1),np.uint8)
        self.Area = abs((self.Koordinat1.x-self.Koordinat2.x)*(self.Koordinat1.y-self.Koordinat2.y))
        cv2.rectangle(self.Maske,(Koordinat1.x,Koordinat1.y),(Koordinat2.x,Koordinat2.y),255,cv2.FILLED)

    def changeStatus(self,bool):
        self.Status = bool


def main():
    video_r = cv2.VideoCapture("Relaxing highway traffic.mp4")
    fgbg = cv2.createBackgroundSubtractorMOG2()

    Sensor1 = Sensör(Koordinat(140,180),Koordinat(255,240),250,1080)
    Sensor2 = Sensör(Koordinat(330,180),Koordinat(420,240),250,1080)
    Sensor3 = Sensör(Koordinat(650,180),Koordinat(730,240),250,1080)
    Sensor4 = Sensör(Koordinat(820,180),Koordinat(910,240),250,1080)
    Sensor_vector = [Sensor1, Sensor2, Sensor3, Sensor4]
    ctr = [0] * len(Sensor_vector)

    while(1):
        ret, Kare = video_r.read()

        cropped_Kare = Kare[350:600,100:1180]
        mask = fgbg.apply(cropped_Kare)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        cleared_im = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        __, cleared_im = cv2.threshold(cleared_im, 127, 255, cv2.THRESH_BINARY)

        sonuc = cropped_Kare.copy()
        black_image = np.zeros((cropped_Kare.shape[0],cropped_Kare.shape[1],1),np.uint8)

        contours,___ = cv2.findContours(cleared_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if w>20 and h>30:

                cv2.rectangle(sonuc,(x,y),(x+w,y+h),(0,255,0),4)
                cv2.rectangle(black_image,(x,y),(x+w,y+h),(255),thickness=cv2.FILLED)


        Sensor2mask_sonuc = []
        Sensor2Oran = []

        for i in range(4):
            Sensor2mask_sonuc.append(cv2.bitwise_and(black_image, black_image, mask=Sensor_vector[i].Maske))
            Sensor2sum = np.sum(Sensor2mask_sonuc[i] == 255)
            Sensor2Oran.append(Sensor2sum / Sensor_vector[i].Area)

        for i in range(4):
            if Sensor2Oran[i] >= 0.75:
                if Sensor_vector[i].Status==False:
                    cv2.rectangle(sonuc,(Sensor_vector[i].Koordinat1.x,Sensor_vector[i].Koordinat1.y),(Sensor_vector[i].Koordinat2.x,Sensor_vector[i].Koordinat2.y),(0,255,0),-1)
                    Sensor_vector[i].changeStatus(True)

                    ctr[i]+=1

                else:
                    cv2.rectangle(sonuc,(Sensor_vector[i].Koordinat1.x,Sensor_vector[i].Koordinat1.y),(Sensor_vector[i].Koordinat2.x,Sensor_vector[i].Koordinat2.y),(0,255,0),-1)

            else:
                cv2.rectangle(sonuc,(Sensor_vector[i].Koordinat1.x,Sensor_vector[i].Koordinat1.y),(Sensor_vector[i].Koordinat2.x,Sensor_vector[i].Koordinat2.y),(0,0,255),-1)
                Sensor_vector[i].changeStatus(False)

        for i in range(4):
            cv2.putText(sonuc, str(ctr[i]), (Sensor_vector[i].Koordinat1.x, Sensor_vector[i].Koordinat1.y + 60),cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255))

        cv2.imshow("rectangle",sonuc)

        k=cv2.waitKey(30) & 0xff

        if k==27:
            break

    video_r.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
