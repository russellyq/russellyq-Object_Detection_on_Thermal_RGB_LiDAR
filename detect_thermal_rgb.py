import argparse
import os
import sys
from pathlib import Path
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sys import platform
import cv2
from models import *  # set ONNX_EXPORT in models.py
# from utils.datasets import *
from utils.utils import *
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import message_filters
# yolov5
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
sys.path.append('/home/yq-robot/github_ws/')

def nms(preds, iou_thresh=0.5):
    # preds: bbox, conf, cls
    # print(np.array(preds[:, 0]))
    bboxes, scores = [], []
    for i in preds:
        bboxes.append([i[0][0], i[0][1], i[0][2], i[0][3]])
        scores.append(i[1])
    # bboxes = np.array(preds[:, 0]).reshape((-1, 4))
    bboxes = np.array(bboxes).reshape((-1, 4))
    scores = np.array(bboxes).reshape((-1, 1))
    """
    :param bboxes: 检测框列表
    :param scores: 置信度列表
    :param iou_thresh: IOU阈值
    :return:
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)

    # 结果列表
    result = []
    index = scores.argsort()[::-1]  # 对检测框按照置信度进行从高到低的排序，并获取索引
    # 下面的操作为了安全，都是对索引处理
    while index.size > 0:
        # 当检测框不为空一直循环
        i = index[0]
        result.append(i)  # 将置信度最高的加入结果列表

        # 计算其他边界框与该边界框的IOU
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        # 只保留满足IOU阈值的索引
        idx = np.where(ious <= iou_thresh)[0]
        index = index[idx + 1]  # 处理剩余的边框
    # bboxes, scores = bboxes[result], scores[result]
    return preds[result]

def iou_batch(boxA, boxB):
    
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def letterbox(img, new_shape=(416, 416), color=(128, 128, 128),
              auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

class Detector(object):
    def __init__(self, opt) -> None:
        
        self.H = np.array([ 6.9303534671360767e-01, -4.8493924029655664e-03,
                            1.1418921832085816e+02, 9.7688882144623111e-04,
                            7.0913456308271539e-01, 3.4890281087940735e+01,
                            -3.2991301283395831e-05, 3.2517689135438541e-05, 1.]).reshape((3,3))
        super().__init__()
        self.opt = opt
        self.bridge = CvBridge()
        self.my_img = None
        self.vid_writer = None
        self.number = 0
        
        self.init_models()
        
        rostopic_lists = ['/thermal_cam/thermal_image/compressed',
                          '/thermal_cam/thermal_image',
                          '/rgb_cam/image_raw/compressed']
        
        rospy.init_node('yolov5_1', anonymous=True)
        
        self.thermal_sub = message_filters.Subscriber(rostopic_lists[0], CompressedImage)
        self.rgb_sub = message_filters.Subscriber(rostopic_lists[2], CompressedImage)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.thermal_sub, self.rgb_sub], 1, 0.1)
        self.ts.registerCallback(self.Callback)
        
        # self.thermal_sub = rospy.Subscriber(rostopic_lists[0], CompressedImage, callback=self.thermal_callback, queue_size=1)
        # self.rgb_sub = rospy.Subscriber(rostopic_lists[2], CompressedImage, callback=self.rgb_callback, queue_size=1)
        
        self.thermal_pub = rospy.Publisher('/thermal_cam/detected_img', Image)
        self.rgb_pub = rospy.Publisher('/rgb_cam/detected_img', Image)
        
        self.project_img_pub = rospy.Publisher('/detected_img_projection', Image)
        self.fused_img_pub = rospy.Publisher('/detected_img_fusion', Image)
    
    def init_models(self):
        
        self.img_size = (320, 192) if ONNX_EXPORT else self.opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        
        # Initialize
        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else self.opt.device)
        
        # Initialize models
        self.model_rgb = Darknet(self.opt.cfg_rgb, self.img_size)
        self.model_thermal = Darknet(self.opt.cfg_thermal, self.img_size)
        
        # Load weights
        # attempt_download(self.opt.weights)
        if self.opt.weights_rgb.endswith('.pt'):  # pytorch format
            self.model_rgb.load_state_dict(torch.load(self.opt.weights_rgb, map_location=self.device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(self.model_rgb, self.opt.weights_rgb)
        
        if self.opt.weights_thermal.endswith('.pt'):  # pytorch format
            self.model_thermal.load_state_dict(torch.load(self.opt.weights_thermal, map_location=self.device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(self.model_thermal, self.opt.weights_thermal)
        
        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model'])  # load weights
            self.modelc.to(self.device).eval()
        # Eval mode
        self.model_rgb.to(self.device).eval()
        self.model_thermal.to(self.device).eval()

        # # Export mode
        # if ONNX_EXPORT:
        #     img = torch.zeros((1, 3) + self.img_size)  # (1, 3, 320, 192)
        #     torch.onnx.export(self.model, img, 'weights/export.onnx', verbose=False, opset_version=11)

        #     # Validate exported model
        #     import onnx
        #     self.model = onnx.load('weights/export.onnx')  # Load the ONNX model
        #     onnx.checker.check_model(self.model)  # Check that the IR is well formed
        #     print(onnx.helper.printable_graph(self.model.graph))  # Print a human readable representation of the graph
        #     return
        
        # # Half precision
        # half = self.opt.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        # if half:
        #     self.model.half()

        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        # Get classes and colors
        self.classes_thermal = load_classes(parse_data_cfg(self.opt.data_thermal)['names'])
        self.colors_thermal = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes_thermal))]
        self.classes_rgb = load_classes(parse_data_cfg(self.opt.data_rgb)['names'])
        self.colors_rgb = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes_rgb))]
    
    def whiten(self,img):
        # image whitening
        img = img / 255.0
        mean, std = img.mean(), img.std()
        img = (img - mean) / std
        img = np.clip(img,-1.0,1.0)
        img = (img + 1.0) / 2.0
        img = img * 255
        img = img.astype(np.uint8)
        return mean, std, img
    
    def thermal2rgb(self, thermal_preds):
        therma2rgb_pred = []
        for det in thermal_preds:
            x1, y1, x2, y2 = det[0]
            X1, Y1, _ = np.matmul(self.H, np.array([[x1],[y1],[1]]))
            X2, Y2, _ = np.matmul(self.H, np.array([[x2],[y2],[1]]))
            therma2rgb_pred.append([[X1, Y1, X2, Y2], det[1], det[2]])
        return np.array(therma2rgb_pred)
    
    def rgb2thermal(self, rgb_preds):
        rgb2thermal_pred = []
        for det in rgb_preds:
            x1, y1, x2, y2 = det[0]
            X1, Y1, _ = np.matmul(np.linalg.inv(self.H), np.array([[x1],[y1],[1]]))
            X2, Y2, _ = np.matmul(np.linalg.inv(self.H), np.array([[x2],[y2],[1]]))
            rgb2thermal_pred.append([[X1, Y1, X2, Y2], det[1], det[2]])
        return rgb2thermal_pred
    
    def Callback(self, thermal_msg, rgb_msg):
        t0 = time.time()
        print("Recieve")
        try:
            self.thermal_img = self.bridge.compressed_imgmsg_to_cv2(thermal_msg, 'passthrough')
            self.rgb_img = self.bridge.compressed_imgmsg_to_cv2(rgb_msg, 'passthrough')
            # _, _, self.thermal_img = self.whiten(self.thermal_img)
            # _, _, self.rgb_img = self.whiten(self.rgb_img)
        except CvBridgeError as e:
            print(e)

        with torch.no_grad():
            thermal_im0, thermal_pred = self.detect(self.thermal_img, mode='thermal')
            rgb_im0, rgb_pred = self.detect(self.rgb_img, mode='rgb')
        

        thermal2rgb_pred = self.thermal2rgb(thermal_pred)
        
        # project thermal pred into rgb
        project_img = self.rgb_img.copy()
        for xyxy, conf, cls in reversed(thermal2rgb_pred):
            label = '%s %.2f' % (cls, conf)
            plot_one_box(xyxy, project_img, label=label, color=self.colors_thermal[0])

        # fused thermal pred into rgb 
        fused_pred = self.datafusion2Dand2D(thermal2rgb_pred, rgb_pred)
        fused_img = self.rgb_img.copy()
        for xyxy, conf, cls in reversed(fused_pred):
            label = '%s %.2f' % (cls, conf)
            plot_one_box(xyxy, fused_img, label=label, color=self.colors_rgb[0])
        
        h, w, _ = thermal_im0.shape
        cv2.putText(thermal_im0,'detected:'+str(len(thermal_pred)),(int(0.1*w), int(0.09 * h)),cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
        h, w, _ = rgb_im0.shape
        cv2.putText(rgb_im0,'detected:'+str(len(rgb_pred)),(int(0.1*w), int(0.09 * h)),cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(project_img,'detected:'+str(len(thermal2rgb_pred)),(int(0.1*w), int(0.09 * h)),cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(fused_img,'detected:'+str(len(fused_pred)),(int(0.1*w), int(0.09 * h)),cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
        
        
        try:
            self.rgb_pub.publish(self.bridge.cv2_to_imgmsg(rgb_im0, "passthrough"))
            self.thermal_pub.publish(self.bridge.cv2_to_imgmsg(thermal_im0, "passthrough"))
            
            self.project_img_pub.publish(self.bridge.cv2_to_imgmsg(project_img, "passthrough"))
            self.fused_img_pub.publish(self.bridge.cv2_to_imgmsg(fused_img, "passthrough"))
        except CvBridgeError as e:
            print(e)
        

        print('Done. (%.3fs)' % (time.time() - t0))       

    
    def datafusion2Dand2D(self, thermal_dets, rgb_dets, iou_threshold=0.2, conf_thred=0.5):
        # fuse detection into rgb
        fused_pred = []
        matched_thermal, matched_rgb, unmatched_thermal, unmatched_rgb = [], [], [], [] 
        iou_matrix = np.zeros((len(thermal_dets), len(rgb_dets)), dtype=np.float32)
        for d1, thermal_det in enumerate(thermal_dets):
            for d2, rgb_det in enumerate(rgb_dets):
                if thermal_det[-1] == rgb_det[-1]:
                    iou_matrix[d1, d2] = iou_batch(thermal_det[0], rgb_det[0])
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a),axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))
        
        # 1st: matched obj
        for idx1, idx2 in matched_indices:
            conf_thermal = thermal_dets[idx1][1]
            conf_rgb = rgb_dets[idx2][1]
            
            # thermal_dets[idx1][1] = rgb_dets[idx2][1] = (thermal_dets[idx1][1]**2 + rgb_dets[idx2][1]**2)/(thermal_dets[idx1][1] + rgb_dets[idx2][1])
            
            if conf_thermal>=conf_rgb:
                fused_pred.append(thermal_dets[idx1])
            else:
                fused_pred.append(rgb_dets[idx2])
            
            matched_thermal.append(thermal_dets[idx1])
            matched_rgb.append(rgb_dets[idx2])    
        
        # 2nd
        for d, det in enumerate(thermal_dets):
            if d not in matched_indices[:, 0] and det[1] >= conf_thred:
                unmatched_thermal.append(det)
                fused_pred.append(det)
        
        for d, det in enumerate(rgb_dets):
            if d not in matched_indices[:, 1] and det[1] >= conf_thred:
                unmatched_rgb.append(det)
                fused_pred.append(det)
        
        return fused_pred
        # return self.rgb2thermal(fused_pred)
    
           

    
    def thermal_callback(self, thermal_msg):
        print("Recieve thermal")
        try:
            self.thermal_img = self.bridge.compressed_imgmsg_to_cv2(thermal_msg, 'passthrough')
        except CvBridgeError as e:
            print(e)
        with torch.no_grad():
            self.detect(self.thermal_img, mode='thermal')

    def rgb_callback(self, rgb_msg):
        print("Recieve rgb")
        try:
            self.rgb_img = self.bridge.compressed_imgmsg_to_cv2(rgb_msg, 'passthrough')
        except CvBridgeError as e:
            print(e)

        with torch.no_grad():
            self.detect(self.rgb_img, mode='rgb')
    
    def detect(self, my_img, mode):
        img, im0 = self.preprocess_img(my_img)
        # Run inference
        
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        if mode == 'thermal':
            pred = self.model_thermal(img)[0]
        
        elif mode == 'rgb':
            pred = self.model_rgb(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.nms_thres, self.opt.classes)
        
        # Apply
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, im0)
        
        prediction = []
        # bbox, conf, cls
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            s = '%g: ' % i

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                if mode == 'thermal':   
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.classes_thermal[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (self.classes_thermal[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=self.colors_thermal[int(cls)])
                        bbox = [i.cpu().numpy() for i in xyxy]
                        prediction.append([bbox, conf.cpu().tolist(), self.classes_thermal[int(cls)]])
                
                elif mode == 'rgb':   
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.classes_rgb[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (self.classes_rgb[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=self.colors_rgb[int(cls)])
                        bbox = [i.cpu().numpy() for i in xyxy]
                        prediction.append([bbox, conf.cpu().tolist(), self.classes_rgb[int(cls)]])
                    
        
        return im0, np.array(prediction)
        
        # if self.opt.save_video:
        #     # vid_path = './video.mp4'
        #     # if isinstance(self.vid_writer, cv2.VideoWriter):
        #     #     self.vid_writer.release()  # release previous video writer

        #     # fps = 10
        #     # h, w, _ = im0.shape
        #     # self.vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        #     # self.vid_writer.write(im0)   
        #     cv2.imwrite('./output/'+str(self.number)+'.png', im0)    
        #     self.number += 1           
            
        
    
    
    def preprocess_img(self, my_img):
        # check for common shapes
        img0 = my_img.copy()
        img = [letterbox(img0, self.img_size, interp=cv2.INTER_LINEAR)[0]]
        img = np.stack(img, 0)
        
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
        return img, img0
        
        

# def detect(save_txt=False, save_img=False):
#     img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
#     out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
#     webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

#     # Initialize
#     device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
#     if os.path.exists(out):
#         shutil.rmtree(out)  # delete output folder
#     os.makedirs(out)  # make new output folder

#     # Initialize model
#     model = Darknet(opt.cfg, img_size)

#     # Load weights
#     attempt_download(weights)
#     if weights.endswith('.pt'):  # pytorch format
#         model.load_state_dict(torch.load(weights, map_location=device)['model'])
#     else:  # darknet format
#         _ = load_darknet_weights(model, weights)

#     # Second-stage classifier
#     classify = False
#     if classify:
#         modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
#         modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
#         modelc.to(device).eval()

#     # Fuse Conv2d + BatchNorm2d layers
#     # model.fuse()

#     # Eval mode
#     model.to(device).eval()

#     # Export mode
#     if ONNX_EXPORT:
#         img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
#         torch.onnx.export(model, img, 'weights/export.onnx', verbose=False, opset_version=11)

#         # Validate exported model
#         import onnx
#         model = onnx.load('weights/export.onnx')  # Load the ONNX model
#         onnx.checker.check_model(model)  # Check that the IR is well formed
#         print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
#         return

#     # Half precision
#     half = half and device.type != 'cpu'  # half precision only supported on CUDA
#     if half:
#         model.half()

#     # Set Dataloader
#     vid_path, vid_writer = None, None
#     if webcam:
#         view_img = True
#         torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
#         dataset = LoadStreams(source, img_size=img_size, half=half)
#     else:
#         save_img = True
#         dataset = LoadImages(source, img_size=img_size, half=half)

#     # Get classes and colors
#     classes = load_classes(parse_data_cfg(opt.data)['names'])
#     colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

#     # Run inference
#     t0 = time.time()
#     for path, img, im0s, vid_cap in dataset:
#         t = time.time()

#         # Get detections
#         img = torch.from_numpy(img).to(device)
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#         pred = model(img)[0]

#         if opt.half:
#             pred = pred.float()

#         # Apply NMS
#         pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

#         # Apply
#         if classify:
#             pred = apply_classifier(pred, modelc, img, im0s)

#         # Process detections
#         for i, det in enumerate(pred):  # detections per image
#             if webcam:  # batch_size >= 1
#                 p, s, im0 = path[i], '%g: ' % i, im0s[i]
#             else:
#                 p, s, im0 = path, '', im0s

#             save_path = str(Path(out) / Path(p).name)
#             s += '%gx%g ' % img.shape[2:]  # print string
#             if det is not None and len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

#                 # Print results
#                 for c in det[:, -1].unique():
#                     n = (det[:, -1] == c).sum()  # detections per class
#                     s += '%g %ss, ' % (n, classes[int(c)])  # add to string

#                 # Write results
#                 for *xyxy, conf, _, cls in det:
#                     if save_txt:  # Write to file
#                         with open(save_path + '.txt', 'a') as file:
#                             file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

#                     if save_img or view_img:  # Add bbox to image
#                         label = '%s %.2f' % (classes[int(cls)], conf)
#                         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

#             print('%sDone. (%.3fs)' % (s, time.time() - t))

#             # Stream results
#             if view_img:
#                 cv2.imshow(p, im0)

#             # Save results (image with detections)
#             if save_img:
#                 if dataset.mode == 'images':
#                     cv2.imwrite(save_path, im0)
#                 else:
#                     if vid_path != save_path:  # new video
#                         vid_path = save_path
#                         if isinstance(vid_writer, cv2.VideoWriter):
#                             vid_writer.release()  # release previous video writer

#                         fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
#                     vid_writer.write(im0)

#     if save_txt or save_img:
#         print('Results saved to %s' % os.getcwd() + os.sep + out)
#         if platform == 'darwin':  # MacOS
#             os.system('open ' + out + ' ' + save_path)

#     print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_rgb', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path for rgb')
    parser.add_argument('--cfg_thermal', type=str, default='cfg/yolov3-spp-r.cfg', help='cfg file path for thermal')
    parser.add_argument('--data_thermal', type=str, default='data/custom.data', help='coco.data file path for thermal')
    parser.add_argument('--data_rgb', type=str, default='data/coco.data', help='coco.data file path for rgb')
    parser.add_argument('--weights_rgb', type=str, default='weights/yolov3-spp.weights', help='path to weights file for rgb')
    parser.add_argument('--weights_thermal', type=str, default='weights/final.pt', help='path to weights file for thermal')
    # parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    # parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save_video', action='store_true', help='save vidoe')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    opt = parser.parse_args()
    print(opt)

    detector = Detector(opt)
    rospy.spin()
