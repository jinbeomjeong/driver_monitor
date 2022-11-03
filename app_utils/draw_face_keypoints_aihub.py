import cv2
import os
from read_json import read_key_points


image_path = './dataset/training/image_semi_constrain/Q_006_20_F_01_M0_G0_C0'
images_name = os.listdir(image_path)
images_name.sort()

label_path = './dataset/training/label_semi_constrain/Q_006_20_F_01_M0_G0_C0'
labels_name = os.listdir(label_path)
labels_name.sort()

for label in labels_name:
    path = os.path.join(image_path, label.rstrip('json')+'jpg')
    frame = cv2.imread(path)
    key_points = read_key_points(os.path.join(label_path, label))

    for i, (x, y) in enumerate(zip(key_points[0:136:2], key_points[1:138:2])):
        cv2.circle(frame, (int(float(x)), int(float(y))), 3, (255, 0, 0), -1)
        cv2.putText(frame, str(i), (int(float(x)), int(float(y))), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(-1) & 0xFF == ord('q'): break

