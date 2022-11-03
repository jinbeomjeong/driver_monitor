import cv2
import numpy as np
# import cupy as cp


video_path = './PIPNet/videos/night.mp4'
video = cv2.VideoCapture(video_path)

while video.isOpened():
    ret, frame = video.read()

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        dst = cv2.equalizeHist(v)

        hsv_dst = np.zeros([720, 1280, 3], dtype=np.uint8).reshape([-1, 3])

        h = h.reshape(-1)
        s = s.reshape(-1)
        dst = dst.reshape(-1)

        # hsv_dst = np.array(np.array([0, 0, 0]), dtype=np.uint8)

        for i in range(hsv_dst.shape[0]):
            hsv_dst[i] = h[i], s[i], dst[i]

        hsv_dst = hsv_dst.reshape(hsv.shape[0], hsv.shape[1], hsv.shape[2])
        frame_dst = cv2.cvtColor(hsv_dst, cv2.COLOR_HSV2BGR)
        cv2.imshow('video', frame_dst)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    else:
        break


