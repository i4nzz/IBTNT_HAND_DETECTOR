import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os



cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# PASTA DOS DADOS COLETADOS
folder = ""  
if not os.path.exists(folder):
    os.makedirs(folder)

saving = False
frames_saved = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        if saving:
            frames_saved += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(f"Frame {frames_saved} salvo.")

    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord('s'):
        saving = True  # Começar a salvar
        frames_saved = 0
    elif key == ord('e'):  # 'e' para encerrar salvamento
        saving = False
        print("Parou de salvar.")
    elif key == ord('q'):
        break