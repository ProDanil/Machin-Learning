import cv2
import numpy as np

import time
from threading import Thread
import os


def play_voice(p):
    # создать команду для воспроизведения музыки, затем выполнить
    # команда
    command = "aplay -q {}".format(p)
    os.system(command)


CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
config_path = "cfg/yolov3.cfg"
weights_path = "weights/yolov3.weights"
font_scale = 1
thickness = 1
found_obj = {}
LABELS = open("data/coco.names").read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

ln = net.getLayerNames()
try:
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    # in case getUnconnectedOutLayers() returns 1D array when CUDA isn't available
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)

while True:
    _, image = cap.read()

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    # print("Time took:", time_took)
    boxes, confidences, class_ids = [], [], []
    pos = 'none'

    # loop over each of the layer outputs
    for output in layer_outputs:
        # перебираем все обнаруженные объекты
        for detection in output:
            # извлекаем идентификатор класса (метку) и достоверность (как вероятность)
            # обнаружение текущего объекта
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # отбросим слабые прогнозы, убедившись, что у обнаруженных
            # вероятность больше минимальной вероятности
            if confidence > CONFIDENCE:
                # масштабируем координаты ограничивающего прямоугольника относительно
                # размер изображения, учитывая, что YOLO на самом деле
                # возвращает центральные координаты (x, y) ограничивающего
                # поля, за которым следуют ширина и высота полей
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                # используем центральные координаты (x, y) для получения вершины и
                # и левый угол ограничительной рамки
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # обновить наш список координат ограничивающего прямоугольника, достоверности,
                # и идентификаторы класса
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # выполнить не максимальное подавление с учетом оценок, определенных ранее
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

    font_scale = 1
    thickness = 1

    # убедитесь, что существует хотя бы один обнаруженный объект
    if len(idxs) > 0:
        # перебираем сохраняемые индексы
        for i in idxs.flatten():
            # извлекаем координаты ограничивающего прямоугольника
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            if x + w / 2 <= 200:
                pos = 'left'
            elif 200 < x + w / 2 <= 400:
                pos = 'front'
            else:
                pos = 'right'

            # рисуем прямоугольник ограничивающей рамки и подписываем на изображении
            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            label = LABELS[class_ids[i]] + ' ' + pos
            text = f"{label}: {confidences[i]:.2f}"
            # print(text)
            found_obj[label] = [found_obj.get(label, [0, True])[0] + 1, True]
            print(found_obj)
            # вычисляем ширину и высоту текста, чтобы рисовать прозрачные поля в качестве фона текста
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                        fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2,
                                                           text_offset_y - text_height))
            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            # добавить непрозрачность (прозрачность поля)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            # теперь поместите текст (метка: доверие%)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
    for key, val in found_obj.items():
        if not val[1]:
            found_obj[key] = [0, False]
        else:
            found_obj[key][1] = False
        if val[0] == 10:
            musicThread = Thread(target=play_voice,
                                 args=(f"voices/{key.split()[0]}.wav",))
            musicThread.daemon = False
            musicThread.start()
            musicThread.join()
            musicThread = Thread(target=play_voice,
                                 args=(f"voices/{key.split()[-1]}.wav",))
            musicThread.daemon = False
            musicThread.start()
            musicThread.join()

    cv2.imshow("image", image)
    if ord("q") == cv2.waitKey(1):
        break

cap.release()
cv2.destroyAllWindows()
