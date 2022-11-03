import cv2
import os
from pts_loader import load


path = './dataset/300W/02_Outdoor'
file_names = os.listdir(path)
file_names.sort()

for file_name in file_names:
    name, ext_name = file_name.split('.')[0], file_name.split('.')[1]

    if 'png' == ext_name:
        frame = cv2.imread(os.path.join(path, file_name))
        points = load(os.path.join(path, name + '.pts'))

        for i, point in enumerate(points):
            cv2.putText(frame, str(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(-1) & 0xFF == ord('q'): break

