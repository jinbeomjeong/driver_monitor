import cv2
import os
import sys
import numpy as np
import pickle
import importlib
import threading
from math import floor
from PIPNet.FaceBoxesV2.faceboxes_detector import *
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
# from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIPNet.lib.networks import *
import PIPNet.lib.data_utils
from PIPNet.lib.functions import *
from PIPNet.lib.mobilenetv3 import mobilenetv3_large
from L2CS_Net.model import L2CS
from L2CS_Net.utils import draw_gaze
from PIL import Image
from scipy.spatial import distance
import joblib
from pandas import DataFrame
from utils.accessory_lib import system_info

elapsed_time = 0
ref_frame = 0
det_frame = 0
fps = 0

system_info()


class LoggingFile:
    def __init__(self, logging_header, file_name='Logging_Data'):
        start_time = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        logging_file_name = file_name + '_' + start_time
        self.logging_file_path = './logging_data/' + logging_file_name + '.csv'
        logging_header.to_csv(self.logging_file_path, mode='a', header=True)

    def start_logging(self, period=0.1):
        # global logging_data, elapsed_time, ref_frame, det_frame, fps
        logging_data = DataFrame({'1': round(elapsed_time, 2), '2': ref_frame, '3': det_frame, '4': round(fps, 2)}, index=[0])
        logging_data.to_csv(self.logging_file_path, mode='a', header=False)
        logging_thread = threading.Timer(period, self.start_logging, (period, ))
        logging_thread.daemon = True
        logging_thread.start()


input_size = 256
net_stride = 32
num_nb = 10
data_name = 'data_300W'
experiment_name = 'pip_32_16_60_r101_l2_l1_10_1_nb10'
num_lms = 68
enable_gaze = False

transformations = transforms.Compose([transforms.Resize(448), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

print(torch.backends.cudnn.enabled)
# torch.backends.cuda.matmul.allow_tf32 = True

meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('PIPNet/data', data_name,
                                                                                      'meanface.txt'), num_nb)

#resnet18 = models.resnet18(pretrained=True)
#landmark_net = Pip_resnet18(resnet18, num_nb=num_nb, num_lms=98, input_size=input_size, net_stride=net_stride)

resnet101 = models.resnet101(pretrained=True)
landmark_net = Pip_resnet101(resnet101, num_nb=num_nb, num_lms=num_lms, input_size=input_size, net_stride=net_stride)

device = torch.device("cuda")

landmark_net = landmark_net.to(device)
save_dir = os.path.join('PIPNet/snapshots', data_name, experiment_name)
weight_file = os.path.join(save_dir, 'epoch%d.pth' % (60 - 1))
state_dict = torch.load(weight_file, map_location=device)
landmark_net.load_state_dict(state_dict)
landmark_net.eval()

if enable_gaze:
    gaze_net = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)
    gaze_net = gaze_net.to(device)
    saved_state_dict = torch.load("./L2CS_Net/snapshot/L2CSNet_gaze360.pkl", map_location=device)
    gaze_net.load_state_dict(saved_state_dict)
    gaze_net.eval()

softmax = nn.Softmax(dim=1).to(device)
idx_tensor = [idx for idx in range(90)]
idx_tensor = torch.FloatTensor(idx_tensor).to(device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), normalize])

face_detector = FaceBoxesDetector('FaceBoxes', 'PIPNet/FaceBoxesV2/weights/FaceBoxesV2.pth', True, device)

my_thresh = 0.6
det_box_scale = 1.2
video = cv2.VideoCapture('/home/jinbeom/workspace/videos/daylight.mp4')
ret, frame = video.read()
image_height, image_width, _ = frame.shape

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#fps = video.get(cv2.CAP_PROP_FPS)

# record result video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# writer = cv2.VideoWriter('.' + os.sep + 'night.mp4', fourcc, fps, (frame_width, frame_height))
#model = joblib.load('./model.pkl')


# logging file threading start
logging_header = DataFrame(columns=['Time(sec)', 'Ref_Frame', 'Det_Frame', 'ProcessSpeed(FPS)'])
logging_task = LoggingFile(logging_header, file_name='Logging_Data')
logging_task.start_logging(period=0.1)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
det_frame = 0

while video.isOpened():
    start_time = time.time()
    ret, frame = video.read()

    if ret:
        available_frame = 0
        ref_frame += 1
        detections, _ = face_detector.detect(frame, my_thresh, 1)

        for i in range(len(detections)):
            det_xmin = detections[i][2]
            det_ymin = detections[i][3]
            det_width = detections[i][4]
            det_height = detections[i][5]
            det_xmax = det_xmin + det_width - 1
            det_ymax = det_ymin + det_height - 1

            det_xmin -= int(det_width * (det_box_scale - 1) / 2)
            det_ymin += int(det_height * (det_box_scale - 1) / 2)
            det_xmax += int(det_width * (det_box_scale - 1) / 2)
            det_ymax += int(det_height * (det_box_scale - 1) / 2)
            det_xmin = max(det_xmin, 0)
            det_ymin = max(det_ymin, 0)
            det_xmax = min(det_xmax, image_width - 1)
            det_ymax = min(det_ymax, image_height - 1)
            det_width = det_xmax - det_xmin + 1
            det_height = det_ymax - det_ymin + 1
            det_crop = frame[det_ymin:det_ymax, det_xmin:det_xmax, :]
            det_crop = cv2.resize(det_crop, (input_size, input_size))
            inputs = Image.fromarray(det_crop[:, :, ::-1].astype('uint8'), 'RGB')
            inputs = preprocess(inputs).unsqueeze(0)
            inputs = inputs.to(device)
            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(landmark_net, inputs,
                                                                                                     preprocess, input_size,
                                                                                                     net_stride, num_nb)
            lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
            tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(num_lms, max_len)
            tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(num_lms, max_len)
            tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
            tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
            lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
            lms_pred = lms_pred.cpu().numpy()
            lms_pred_merge = lms_pred_merge.cpu().numpy()

            eye_x = (lms_pred_merge[36 * 2:48 * 2:2] * det_width).astype(np.int32) + det_xmin
            eye_y = (lms_pred_merge[(36 * 2) + 1:(48 * 2) + 1:2] * det_height).astype(np.int32) + det_ymin

            left_eye_H_dist = distance.euclidean((eye_x[0], eye_y[0]), (eye_x[3], eye_y[3]))
            left_eye_V_dist = distance.euclidean((eye_x[1], eye_y[1]), (eye_x[5], eye_y[5])) + \
                              distance.euclidean((eye_x[2], eye_y[2]), (eye_x[4], eye_y[4]))

            right_eye_H_dist = distance.euclidean((eye_x[6], eye_y[6]), (eye_x[9], eye_y[9]))
            right_eye_V_dist = distance.euclidean((eye_x[11], eye_y[11]), (eye_x[7], eye_y[7])) + \
                               distance.euclidean((eye_x[10], eye_y[10]), (eye_x[8], eye_y[8]))

            left_eye_ratio = left_eye_V_dist / left_eye_H_dist
            right_eye_ratio = right_eye_V_dist / right_eye_H_dist
            #result = model.predict(np.array([left_eye_H_dist, left_eye_V_dist, right_eye_H_dist, right_eye_V_dist]).reshape(1, -1), num_iteration=model.best_iteration_)


            #result_color = (255, 255, 255) if result == True else (0, 0, 255)
            cv2.rectangle(frame, (det_xmin, det_ymin), (det_xmax, det_ymax), (255, 255, 255), 2)

            eye_det = 0.35
            left_eye_color = (0, 255, 0) if left_eye_ratio >= eye_det else (0, 0, 255)
            right_eye_color = (0, 255, 0) if right_eye_ratio >= eye_det else (0, 0, 255)

            for i in range(len(eye_x)):
                if i <= 5:
                    cv2.circle(frame, (eye_x[i], eye_y[i]), 1, left_eye_color, 2)
                else:
                    cv2.circle(frame, (eye_x[i], eye_y[i]), 1, right_eye_color, 2)

            if left_eye_ratio > eye_det and right_eye_ratio > eye_det:
                img = frame[det_ymin:det_ymax, det_xmin:det_xmax]
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                img = transformations(im_pil)
                img = Variable(img).to(device)
                img = img.unsqueeze(0)
                det_frame += 1

                # gaze prediction
                if enable_gaze:
                    gaze_pitch, gaze_yaw = gaze_net(img)
                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)

                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180

                    pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
                    yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0
                    draw_gaze(det_xmin, det_ymin, det_width, det_height, frame, (pitch_predicted, yaw_predicted), color=(0, 0, 255))

            cv2.putText(frame, f'{left_eye_ratio:.2f}', (det_xmin, det_ymax+30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.putText(frame, f'{right_eye_ratio:.2f}', (det_xmax-50, det_ymax + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 20), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Raw Frame : ' + '%.0f' % ref_frame, (5, 40), font, 1, [0, 0, 255], 1,  cv2.LINE_AA)
        cv2.putText(frame, 'Det. Frame : ' + '%.0f' % det_frame, (5, 60), font, 1, [0, 0, 255], 1, cv2.LINE_AA)

        # cv2.imwrite('images/1_out.jpg', image)
        # writer.write(frame)
        cv2.imshow('frame', frame)

    else: break

    if cv2.waitKey(1) & 0xFF == ord('q'): break

video.release()
cv2.destroyAllWindows()
print('Video Play Done')
