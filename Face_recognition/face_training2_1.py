"""
Thanks to 黃建庭老師
https://sites.google.com/site/zsgititit/home/raspberry-shu-mei-pai/raspberry3shi-yongwebcam-opencv-jin-xing-ren-lian-bian-shi

I fixed some parts to make it run in current environment.
"""

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file         :face_training2_1.py
@explain      :take picture and save to images
@date         :2023/03/02 13:36:13
@author       :gostjoke
@version      :1.0
'''
# python 3.10
# opencv-contrib-python        4.7.0.72
# opencv-python-headless       4.6.0.66

import cv2
import os

# if no folder
try:
    os.mkdir("images")
except:
    pass
try:
    os.mkdir("train")
except:
    pass

camema = cv2.VideoCapture(0)  #對應/dev/video0的攝影機
camema.set(3, 640) # 設定影片寬度
camema.set(4, 480) # 設定影片高度
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
id = input('\n請輸入id: ') # please input int id
print("\n 初始化錄影機，請等待")
count = 0
while(True):
    ret, img = camema.read()
    img = cv2.flip(img, 1) #設定影像左右互換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #轉換成灰階
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)  #辨識影像
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)     #加上綠框
        count += 1
        cv2.imwrite("images/User." + str(id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])  #儲存影像到dataset資料夾
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff #等待0.1秒，偵測鍵盤按鍵是否按下
    if k == 27:#按下ESC按鍵，中斷while迴圈
        break
    elif count >= 30: # 偵測30張臉後，中斷while迴圈
         break
print("\n 偵測完成")
camema.release()
cv2.destroyAllWindows()