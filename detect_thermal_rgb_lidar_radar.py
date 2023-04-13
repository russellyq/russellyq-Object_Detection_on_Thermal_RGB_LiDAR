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
from models import *
from utils.torch_utils import fuse_conv_and_bn  # set ONNX_EXPORT in models.py
# from utils.datasets import *
from utils.utils import *
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointCloud
import sensor_msgs.point_cloud2 as pcl2
import message_filters
# yolov5
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
sys.path.append('/home/yq-robot/github_ws/')

H, W = 480, 640

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
        self.lidar_depth_img, self.radar_depth_img = None, None
        self.P_rgb = np.array([[332.232689, 0.000000, 333.058485, 0], 
                                    [0, 332.644823, 240.998586, 0], 
                                    [0, 0, 1, 0]])
        self.H = np.array([ 6.9303534671360767e-01, -4.8493924029655664e-03,
                            1.1418921832085816e+02, 9.7688882144623111e-04,
                            7.0913456308271539e-01, 3.4890281087940735e+01,
                            -3.2991301283395831e-05, 3.2517689135438541e-05, 1.]).reshape((3,3))
        self.LiDAR2rgb = np.array([-0.01807319, -0.99976844,  0.01174591, -0.08543468,
                                    -0.03244301, -0.01115523, -0.99940997,  0.11639403,
                                    0.99931043, -0.01844365, -0.03223318, -0.04211223]).reshape((3,4))
        self.rgb2Radar = np.array([-0.02236591803543918, 0.01373899491374072, 0.9996551471376557, 0.04156851224667894,
                                    -0.9997463777260759, -0.003016115343759606, -0.02232651169926307,  0.06252061356319374,
                                    0.002708378892805941, -0.9999009961676391, 0.01380290382175741, 0.06251000189496365]).reshape((3,4))
        self.Radar2rgb = np.zeros((3, 4))
        self.Radar2rgb[0:3, 0:3] = np.linalg.inv(self.rgb2Radar[0:3, 0:3])
        self.Radar2rgb[0:3, -1] = - np.matmul(self.Radar2rgb[0:3, 0:3], self.rgb2Radar[0:3, -1])
        
        super().__init__()
        self.opt = opt
        self.bridge = CvBridge()
        self.number = 0
        
        self.init_models()
        
        rostopic_lists = ['/thermal_cam/thermal_image/compressed',
                          '/rgb_cam/image_raw/compressed',
                          '/livox/lidar/time_sync',
                          '/radar_pcl']
        
        rospy.init_node('yolov5_1', anonymous=True)
        
        self.thermal_sub = message_filters.Subscriber(rostopic_lists[0], CompressedImage)
        self.rgb_sub = message_filters.Subscriber(rostopic_lists[1], CompressedImage)
        
        self.lidar_sub = message_filters.Subscriber(rostopic_lists[2], PointCloud2)
        self.radar_sub = message_filters.Subscriber(rostopic_lists[3], PointCloud)
        
        self.lidar_depth_ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.lidar_sub], 1, 0.1)
        self.lidar_depth_ts.registerCallback(self.lidar_depth_Callback)
        
        self.radar_depth_ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.radar_sub], 1, 0.1)
        self.radar_depth_ts.registerCallback(self.radar_depth_Callback)    
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.thermal_sub, self.rgb_sub, self.lidar_sub, self.radar_sub], 1, 0.1)
        self.ts.registerCallback(self.Callback)

        
        self.thermal_pub = rospy.Publisher('/thermal_cam/detected_img', Image)
        self.rgb_pub = rospy.Publisher('/rgb_cam/detected_img', Image)
        
        self.project_img_pub = rospy.Publisher('/detected_img_projection', Image)
        self.fused_img_pub = rospy.Publisher('/detected_img_fusion', Image)
        self.fused_img_LiDAR_pub = rospy.Publisher('/detected_img_fusion_LiDAR', Image)
        self.fused_img_Radar_pub = rospy.Publisher('/detected_img_fusion_Radar', Image)
        rospy.spin()

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
            X1, Y1, _ = np.matmul(self.H, np.array([[x1],[y1],[1]])).reshape((1, 3)).astype(np.int16).tolist()[0]
            X2, Y2, _ = np.matmul(self.H, np.array([[x2],[y2],[1]])).reshape((1, 3)).astype(np.int16).tolist()[0]
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
    
    def Callback(self, thermal_msg, rgb_msg, lidar_msg, radar_msg):
        if self.radar_depth_img is None or self.lidar_depth_img is None: 
            return
        t0 = time.time()
        print("Recieve")
        try:
            self.thermal_img = self.bridge.compressed_imgmsg_to_cv2(thermal_msg, 'passthrough')
            self.rgb_img = self.bridge.compressed_imgmsg_to_cv2(rgb_msg, 'passthrough')
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
        
        fused_img_LiDAR, fused_img_Radar = self.rgb_img.copy(), self.rgb_img.copy()
        
        # projectiion into image
        fused_img_LiDAR, _ = self.lidar_radar_projection(fused_pred, fused_img_LiDAR, 'lidar')
        fused_img_Radar, _ = self.lidar_radar_projection(fused_pred, fused_img_Radar, 'radar')

        
        h, w, _ = thermal_im0.shape
        cv2.putText(thermal_im0,'detected:'+str(len(thermal_pred)),(int(0.1*w), int(0.09 * h)),cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
        h, w, _ = rgb_im0.shape
        cv2.putText(rgb_im0,'detected:'+str(len(rgb_pred)),(int(0.1*w), int(0.09 * h)),cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(project_img,'detected:'+str(len(thermal2rgb_pred)),(int(0.1*w), int(0.09 * h)),cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
        
        cv2.putText(fused_img_LiDAR,'detected:'+str(len(fused_pred)),(int(0.1*w), int(0.09 * h)),cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(fused_img_Radar,'detected:'+str(len(fused_pred)),(int(0.1*w), int(0.09 * h)),cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
        
        try:
            self.rgb_pub.publish(self.bridge.cv2_to_imgmsg(rgb_im0, "passthrough"))
            self.thermal_pub.publish(self.bridge.cv2_to_imgmsg(thermal_im0, "passthrough"))
            
            self.project_img_pub.publish(self.bridge.cv2_to_imgmsg(project_img, "passthrough"))
            # self.fused_img_pub.publish(self.bridge.cv2_to_imgmsg(fused_img, "passthrough"))
            self.fused_img_LiDAR_pub.publish(self.bridge.cv2_to_imgmsg(fused_img_LiDAR, "passthrough"))
            self.fused_img_Radar_pub.publish(self.bridge.cv2_to_imgmsg(fused_img_Radar, "passthrough"))

        except CvBridgeError as e:
            print(e)
        
        print('Done: (%.2fs)' % (time.time() - t0))
    

    def lidar_depth_Callback(self, img_msg, lidar_msg):
        t0 = time.time()
        
        points_list = []
        for point in pcl2.read_points(lidar_msg, skip_nans=True, field_names=("x", "y", "z")):
            if point[0] == 0 and point[1] == 0 and point[2] == 0:
                continue
            points_list.append([point[0], point[1], point[2]])
        
        points_list = np.asarray(points_list)
        points_dis = np.linalg.norm(points_list, axis=1)
        
        points_3d_cam = self.lidar_radar_to_camera(points_list, self.LiDAR2rgb)
        points_2d = self.project_to_image(points_3d_cam, self.P_rgb)
        
        points2d_dis = np.concatenate([points_2d, points_dis.reshape((-1, 1))], axis=1)
        
        depth_img = np.zeros((H,W))
        for u,v,dis in points2d_dis:
            if u >= 0 and u < W and v >= 0 and v < H:
                depth_img[int(v),int(u)] = dis
        
        self.lidar_depth_img = depth_img
        print('Publishing LiDAR Depth Image: (%.2fs)' % (time.time() - t0))

    def radar_depth_Callback(self, img_msg, radar_msg):
        t0 = time.time()
        
        points_list = read_radar_points(radar_msg)
        points_dis = np.linalg.norm(points_list, axis=1)
        
        points_3d_cam = self.lidar_radar_to_camera(points_list, self.Radar2rgb)
        points_2d = self.project_to_image(points_3d_cam, self.P_rgb)
        
        points2d_dis = np.concatenate([points_2d, points_dis.reshape((-1, 1))], axis=1)
        
        depth_img = np.zeros((H,W))
        for u,v,dis in points2d_dis:
            if u >= 0 and u < W and v >= 0 and v < H:
                depth_img[int(v),int(u)] = dis
        
        self.radar_depth_img = depth_img
        print('Publishing Radar Depth Image: (%.2fs)' % (time.time() - t0))

    def lidar_radar_projection(self, fused_pred, fused_img, mode='lidar'):   
        depth_img = self.lidar_depth_img if mode=='lidar' else self.radar_depth_img

        # dis_max, dis_min = np.max(points2d_dis[:, -1]), np.min(points2d_dis[:, -1])
        
        # for u, v, dis in points2d_dis:
        #     if 0<u<W and 0<v<H:
        #         fused_img[int(v), int(u), 0] = 255 - (dis-dis_min)/(dis_max-dis_min) * 255
        #         fused_img[int(v), int(u), 1] = 255 - (dis-dis_min)/(dis_max-dis_min) * 255
        #         fused_img[int(v), int(u), 2] = 255 - (dis-dis_min)/(dis_max-dis_min) * 255

        for det in fused_pred:
            x1, y1, x2, y2 = det[0]
            
            dis_matrix = depth_img[int(y1):int(y2), int(x1):int(x2)].reshape((1,-1))
            dis_matrix = dis_matrix[np.where(dis_matrix > 0)]
            
            det[-2] = np.median(dis_matrix)          
        
        for xyxy, conf, cls in reversed(fused_pred):
            label = '%.2f' % conf
            plot_one_box(xyxy, fused_img, label=label, color=self.colors_rgb[0])
        
        return fused_img, fused_pred
    
    def lidar_radar_to_camera(self, points, extrinsic_param):
        # points: (8*3)
        points = np.concatenate((points, np.ones((points.shape[0],1))), axis=1)
        p = np.matmul(extrinsic_param, points.T)
        return p.T # 8*3
    
    def project_to_image(self, pts_3d, intrinsic_camera):
        # pts_3d: 8 x 3
        # P: 3 x 4
        # return: 8 x 2
        pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
        pts_2d = np.dot(intrinsic_camera, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
        return pts_2d.astype(np.int32)
            
        
    
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
                        bbox = [i.cpu().numpy().tolist() for i in xyxy]                    
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
                        bbox = [i.cpu().numpy().tolist() for i in xyxy]                        
                        prediction.append([bbox, conf.cpu().tolist(), self.classes_rgb[int(cls)]])
        return im0, np.array(prediction)       
    
    def preprocess_img(self, my_img):
        # check for common shapes
        img0 = my_img.copy()
        img = [letterbox(img0, self.img_size, interp=cv2.INTER_LINEAR)[0]]
        img = np.stack(img, 0)
        
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
        return img, img0
        
        
def read_radar_points(cloud_msg):
    raw_data = cloud_msg.points
    points = []
    for point in raw_data:
        if point.x == 0 and point.y == 0 and point.z == 0:
            continue
        points.append([point.x, point.y, point.z])
    return np.array(points).reshape((-1, 3))
        


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
