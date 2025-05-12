import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf

# Caminhos
tflite_model_path = "C:\\_source\\IBTNT_HAND_DETECTOR\\Treined_Model\\model_unquant.tflite"
labels_path = "C:\\_source\\IBTNT_HAND_DETECTOR\\Treined_Model\\labels.txt"

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(labels_path, "r") as f:
    labels = [line.strip().split(' ')[1] for line in f.readlines()]

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

ultima_letra = ""
texto_final = ""

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[max(0, y-offset): y + h + offset, max(0, x-offset): x + w + offset]
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

        imgInput = cv2.resize(imgWhite, (224, 224))
        imgInput = np.expand_dims(imgInput.astype(np.float32) / 255.0, axis=0)

        interpreter.set_tensor(input_details[0]['index'], imgInput)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        index = np.argmax(prediction)
        letra = labels[index]
        ultima_letra = letra

        # Mostrar letra detectada
        cv2.putText(imgOutput, f"Letra: {letra}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Mostrar palavra sendo formada
    cv2.putText(imgOutput, f"Texto: {texto_final}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    cv2.imshow('Imagem', imgOutput)

    key = cv2.waitKey(1)

    if key == 13:  # Enter
        texto_final += ultima_letra
        print(f"Letra '{ultima_letra}' adicionada!")

    elif key == 32:  # Espaço
        texto_final += ' '

    elif key == 8:  # Backspace
        texto_final = texto_final[:-1]

    elif key == 27:  # ESC
        with open("saida.txt", "w") as f:
            f.write(texto_final)
        print("Texto salvo em 'saida.txt'")
        break

cap.release()
cv2.destroyAllWindows()
