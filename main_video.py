import sys, os, threading, time, torchvision, can, cantools
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from scipy.spatial import distance
from pandas import DataFrame
from app_utils.accessory_lib import pytorch_system_info
from torch.autograd import Variable
from app_utils.tcp_lib import TcpServerCom
from can.interfaces.pcan import PcanBus as pcan

default_path = os.path.normpath(os.path.abspath(__file__)).split(os.sep)[0:-1]
facebox_path = default_path[:]
facebox_path.append('PIPNet/FaceBoxes_PyTorch/')
facebox_path = '/'.join(facebox_path)
sys.path.append(facebox_path)

from models.faceboxes import FaceBoxes
from data import cfg
from utils.box_utils import decode
from utils.nms_wrapper import nms
from layers.functions.prior_box import PriorBox

from PIPNet.lib.networks import *
from PIPNet.lib.functions import *

l2cs_path = default_path
l2cs_path.append('L2CS_Net/')
l2cs_path = '/'.join(l2cs_path)
sys.path.append(l2cs_path)

from L2CS_Net.model import L2CS
from L2CS_Net.utils import draw_gaze


initial_time = time.time()
elapsed_time = 0
ref_frame = 0
det_frame = 0
fps = 0
input_size = 256
net_stride = 32
num_nb = 10
data_name = 'data_300W'
experiment_name = 'pip_32_16_60_r101_l2_l1_10_1_nb10'
num_lms = 68
image_scale = 0.0
offset_height = 0
offset_width = 0
det_box_scale = 1.2
eye_det = 0.15
cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
enable_gaze = True
enable_log = False
enable_tcp = True
enable_record = False
enable_can_com = True

pytorch_system_info()


class LoggingFile:
    def __init__(self, logging_header, file_name='Logging_Data'):
        start_time = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        logging_file_name = file_name + '_' + start_time
        self.logging_file_path = './logging_data/' + logging_file_name + '.csv'
        logging_header.to_csv(self.logging_file_path, mode='a', header=True)


    def start_logging(self, period=0.1):
        # global logging_data, elapsed_time, ref_frame, det_frame, fps
        logging_data = DataFrame({'1': round(elapsed_time, 2), '2': ref_frame, '3': det_frame, '4': round(fps, 2)}, index=[0])
        logging_data.to_csv(self.logging_file_path, mode='a', header=False, index=False)
        logging_thread = threading.Timer(period, self.start_logging, (period, ))
        logging_thread.daemon = True
        logging_thread.start()


transformations = transforms.Compose([transforms.Resize(448), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('PIPNet/data', data_name,
                                                                                      'meanface.txt'), num_nb)
#resnet18 = models.resnet18(pretrained=True)
#landmark_net = Pip_resnet18(resnet18, num_nb=num_nb, num_lms=98, input_size=input_size, net_stride=net_stride)

resnet101 = models.resnet101(weights='ResNet101_Weights.DEFAULT')
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


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
net = load_model(net, 'PIPNet/FaceBoxes_PyTorch/weights/Final_FaceBoxes.pth', False)
net.eval()
net = net.to(device)

# video = cv2.VideoCapture('/mnt/data/video/daylight.mp4')
video = cv2.VideoCapture(0)

cv2.namedWindow('video', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
ret, frame = video.read()

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_width_resize = int((1-image_scale)*frame_width)
frame_height_resize = int((1-image_scale)*frame_height)

scale = torch.Tensor([frame_width_resize, frame_height_resize, frame_width_resize, frame_height_resize])
scale = scale.to(device)

# fps = video.get(cv2.CAP_PROP_FPS)
# record result video

#model = joblib.load('./model.pkl')

# logging task threading start
if enable_log:
    logging_header = DataFrame(columns=['Time(sec)', 'Ref_Frame', 'Det_Frame', 'ProcessSpeed(FPS)'])
    logging_task = LoggingFile(logging_header, file_name='Logging_Data')
    logging_task.start_logging(period=0.1)

if enable_tcp:
    server = TcpServerCom(addr='192.168.137.68', port=6340)

if enable_can_com:
    bus = pcan(bitrate=500000)
    clu_db = cantools.database.load_file('./adas_can_db.dbc').get_message_by_name('CLU_VCU_2A1')

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
det_frame = 0
image_border = image_scale / 2

if enable_record:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('result_video.mp4', fourcc, 30, (frame_width, frame_height))

while video.isOpened():
    start_time = time.time()
    ret, frame = video.read()

    if ret:
        frame = frame[int(frame_height*image_border)+offset_height:int(frame_height-(frame_height*image_border))+offset_height,
                int(frame_width*image_border)+offset_width:int(frame_width-(frame_width*image_border))+offset_width]

        frame = cv2.resize(src=frame, dsize=(frame_width_resize, frame_height_resize), interpolation=cv2.INTER_LINEAR)

        img_tensor = np.float32(frame)
        img_tensor -= (104, 117, 123)
        img_tensor = img_tensor.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        loc, conf = net(img_tensor)

        priorbox = PriorBox(cfg, image_size=(frame_height_resize, frame_width_resize))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        inds = np.where(scores > 0.05)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        order = scores.argsort()[::-1][:5000]

        boxes = boxes[order]
        scores = scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, 0.3, force_cpu=False)
        dets = dets[keep, :]
        dets = dets[:750, :]

        available_frame = 0
        ref_frame += 1

        for i, b in enumerate(dets):
            if b[4] < 0.5:
                continue
            text = "{:.3f}".format(b[4])
            b = list(map(int, b))

            det_xmin = b[0]
            det_ymin = b[1]
            det_xmax = b[2]
            det_ymax = b[3]
            det_width = det_xmax - det_xmin
            det_height = det_ymax - det_ymin

            cv2.rectangle(frame, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 1)
            cv2.putText(frame, text, (det_xmin, det_ymin+12), font, 0.5, (255, 255, 255))

            det_xmin -= int(det_width * (det_box_scale - 1) / 2)
            det_ymin -= int(det_height * (det_box_scale - 1) / 2)
            det_xmax += int(det_width * (det_box_scale - 1) / 2)
            det_ymax += int(det_height * (det_box_scale - 1) / 2)

            det_xmin = max(det_xmin, 0)
            det_ymin = max(det_ymin, 0)
            det_xmax = min(det_xmax, frame_width - 1)
            det_ymax = min(det_ymax, frame_height - 1)

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
            left_eye_V_dist = min(distance.euclidean((eye_x[1], eye_y[1]), (eye_x[5], eye_y[5])),
                                  distance.euclidean((eye_x[2], eye_y[2]), (eye_x[4], eye_y[4])))

            right_eye_H_dist = distance.euclidean((eye_x[6], eye_y[6]), (eye_x[9], eye_y[9]))
            right_eye_V_dist = min(distance.euclidean((eye_x[11], eye_y[11]), (eye_x[7], eye_y[7])),
                                   distance.euclidean((eye_x[10], eye_y[10]), (eye_x[8], eye_y[8])))

            left_eye_ratio = left_eye_V_dist / left_eye_H_dist
            right_eye_ratio = right_eye_V_dist / right_eye_H_dist

            cv2.rectangle(frame, (det_xmin, det_ymin), (det_xmax, det_ymax), (255, 255, 255), 2)

            left_eye_color = (0, 255, 0) if left_eye_ratio >= eye_det else (0, 0, 255)
            right_eye_color = (0, 255, 0) if right_eye_ratio >= eye_det else (0, 0, 255)

            for i in range(len(eye_x)):
                if i <= 5:
                    cv2.circle(frame, (eye_x[i], eye_y[i]), 1, left_eye_color, 2)
                else:
                    cv2.circle(frame, (eye_x[i], eye_y[i]), 1, right_eye_color, 2)

            drowsy_state = left_eye_ratio > eye_det and right_eye_ratio > eye_det

            if enable_can_com:
                dsm_can_payload = clu_db.encode({'dsm': int(left_eye_ratio < eye_det and right_eye_ratio < eye_det)})
                can_message = can.Message(arbitration_id=clu_db.frame_id, is_extended_id=False, data=dsm_can_payload)
                bus.send(can_message)

            if drowsy_state:
                det_frame += 1

                if enable_gaze: # gaze prediction
                    img = frame[det_ymin:det_ymax, det_xmin:det_xmax]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img = transformations(im_pil)
                    img = Variable(img).to(device)
                    img = img.unsqueeze(0)

                    gaze_pitch, gaze_yaw = gaze_net(img)
                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)

                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180

                    pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
                    yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0
                    draw_gaze(det_xmin, det_ymin, det_width, det_height, frame, (pitch_predicted, yaw_predicted), color=(0, 0, 255))
                    tcp_message = f'{pitch_predicted:.2f}' + ',' + f'{yaw_predicted:.2f}'

                    if enable_tcp:
                        server.send_msg(tcp_message)

            cv2.putText(frame, f'{left_eye_ratio:.2f}', (det_xmin, det_ymax+30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'{right_eye_ratio:.2f}', (det_xmax-50, det_ymax + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        fps = 1.0 / (time.time() - start_time)
        elapsed_time = time.time() - initial_time
        #cv2.putText(frame, f'Elapsed time: {elapsed_time:.1f}', (10, 20), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        #cv2.putText(frame, f'FPS: {fps:.1f}', (10, 40), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        #cv2.putText(frame, 'Raw Frame : ' + '%.0f' % ref_frame, (5, 60), font, 1, [0, 0, 255], 1,  cv2.LINE_AA)
        #cv2.putText(frame, 'Det. Frame : ' + '%.0f' % det_frame, (5, 80), font, 1, [0, 0, 255], 1, cv2.LINE_AA)

        # cv2.imwrite('images/1_out.jpg', image)
        # writer.write(frame)
        cv2.imshow('video', frame)

        if enable_record:
            writer.write(frame)

    else: break

    if cv2.waitKey(1) & 0xFF == ord('q'): break

video.release()
cv2.destroyAllWindows()
print('Video Play Done')
