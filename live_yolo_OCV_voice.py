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
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    # print("Time took:", time_took)
    boxes, confidences, class_ids = [], [], []

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
            # рисуем прямоугольник ограничивающей рамки и подписываем на изображении
            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            label = LABELS[class_ids[i]]
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
        if val[0] == 1 or val[0] % 20 == 0:
            musicThread = Thread(target=play_voice,
                                 args=(f"voices/{key}.wav",))
            musicThread.daemon = False
            musicThread.start()

    cv2.imshow("image", image)
    if ord("q") == cv2.waitKey(1):
        break

cap.release()
cv2.destroyAllWindows()


# --------------------------------------------------------------------
# импортируем необходимые пакеты
# from keras.preprocessing.image import img_to_array
# from keras.models import load_model
# from gpiozero import LEDBoard
# from gpiozero.tools import random_values
# from imutils.video import VideoStream
# from threading import Thread
# import numpy as np
# import imutils
# import time
# import cv2
# import os
#
#
# def light_tree(tree, sleep=5):
#     # перебираем все светодиоды в дереве и случайным образом мигаем ими с помощью
#     # различной интенсивности
#     for led in tree:
#         led.source_delay = 0.1
#         led.source = random_values()
#     # sleep for a bit to let the tree show its Christmas spirit for
#     # santa clause
#     time.sleep(sleep)
#     # снова включить светодиоды, на этот раз отключив их
#     for led in tree:
#         led.source = None
#         led.value = 0
#
#
# def play_christmas_music(p):
#     # создать команду для воспроизведения музыки, затем выполнить
#     # команда
#     command = "aplay -q {}".format(p)
#     os.system(command)
#
#
# # определить пути к модели глубокого обучения Not Santa Keras и
# # звуковой файл
# MODEL_PATH = "santa_not_santa.model"
# AUDIO_PATH = "jolly_laugh.wav"
#
# # инициализируем общее количество кадров, которые *последовательно* содержат
# # santa вместе с порогом, необходимым для срабатывания будильника santa
# TOTAL_CONSEC = 0
# TOTAL_THRESH = 20
#
# # инициализировать, если сработала сигнализация Санта-Клауса
# SANTA = False
#
# # загружаем модель
# print("[INFO] loading model...")
# model = load_model(MODEL_PATH)
#
# # initialize the christmas tree
# tree = LEDBoard(*range(2, 28), pwm=True)
#
# # инициализируем видеопоток и даем сенсору камеры прогреться
# print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# # vs = VideoStream(usePiCamera=True).start()
# time.sleep(2.0)
#
# # зацикливаем кадры из видеопотока
# while True:
#     # захватить кадр из потокового видеопотока и изменить его размер
#     # иметь максимальную ширину 400 пикселей
#     frame = vs.read()
#     frame = imutils.resize(frame, width=400)
#
#     # подготовить изображение для классификации нашей сетью глубокого обучения
#     image = cv2.resize(frame, (28, 28))
#     image = image.astype("float") / 255.0
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)
#
#     # классифицировать входное изображение и инициализировать метку и
#     # вероятность предсказания
#     (notSanta, santa) = model.predict(image)[0]
#     label = "Not Santa"
#     proba = notSanta
#
#     # проверить, был ли обнаружен Санта с помощью нашей свертки
#     # neural network
#     if santa > notSanta:
#         # обновить метку и вероятность предсказания
#         label = "Santa"
#         proba = santa
#         # увеличить общее количество последовательных кадров, которые
#         # содержит Санту
#         TOTAL_CONSEC += 1
#
#     # проверяем, должны ли мы поднять тревогу Санта-Клауса
#     if not SANTA and TOTAL_CONSEC >= TOTAL_THRESH:
#         # указывает, что Санта был найден
#         SANTA = True
#
#         # light up the christmas tree
#         treeThread = Thread(target=light_tree, args=(tree,))
#         treeThread.daemon = True
#         treeThread.start()
#
#         # включи рождественские мелодии
#         musicThread = Thread(target=play_christmas_music,
#                              args=(AUDIO_PATH,))
#         musicThread.daemon = False
#         musicThread.start()
#
#     # в противном случае сбрасываем общее количество последовательных кадров и Санта-будильник
#     else:
#         TOTAL_CONSEC = 0
#         SANTA = False
#
#     # создаем метку и рисуем ее на рамке
#     label = "{}: {:.2f}%".format(label, proba * 100)
#     frame = cv2.putText(frame, label, (10, 25),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     # показываем выходной кадр
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#
#     # если была нажата клавиша `q`, выйти из цикла
#     if key == ord("q"):
#         break
#
# # сделать небольшую очистку
# print("[INFO] cleaning up...")
# cv2.destroyAllWindows()
# vs.stop()
