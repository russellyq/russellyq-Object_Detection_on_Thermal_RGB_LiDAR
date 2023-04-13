import argparse
from sys import platform
import cv2
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage

ONNX_EXPORT = False
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
        super().__init__()
        self.opt = opt
        self.bridge = CvBridge()
        self.my_img = None
        self.vid_writer = None
        self.number = 0
        
        self.init_model()
        
        rostopics = ['/thermal_cam/thermal_image/compressed']
        
        rospy.init_node('yolov5_1', anonymous=True)
        
        self.img_sub = rospy.Subscriber(rostopics[0], CompressedImage, callback=self.Callback, queue_size=1)
        self.img_pub = rospy.Publisher('/detected_img', Image)
    
    def init_model(self):
        self.img_size = (320, 192) if ONNX_EXPORT else self.opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        # Initialize
        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else self.opt.device)
        # Initialize model
        self.model = Darknet(self.opt.cfg, self.img_size)
        # Load weights
        attempt_download(self.opt.weights)
        if self.opt.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.opt.weights, map_location=self.device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(self.model, self.opt.weights)
        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model'])  # load weights
            self.modelc.to(self.device).eval()
        # Eval mode
        self.model.to(self.device).eval()
        

        # Export mode
        if ONNX_EXPORT:
            img = torch.zeros((1, 3) + self.img_size)  # (1, 3, 320, 192)
            torch.onnx.export(self.model, img, 'weights/export.onnx', verbose=False, opset_version=11)

            # Validate exported model
            import onnx
            self.model = onnx.load('weights/export.onnx')  # Load the ONNX model
            onnx.checker.check_model(self.model)  # Check that the IR is well formed
            print(onnx.helper.printable_graph(self.model.graph))  # Print a human readable representation of the graph
            return
        
        # Half precision
        half = self.opt.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            self.model.half()

        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        # Get classes and colors
        self.classes = load_classes(parse_data_cfg(self.opt.data)['names'])
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
    
    def Callback(self, img_msg):
        print("Recieve")
        try:
            self.my_img = self.bridge.compressed_imgmsg_to_cv2(img_msg, 'passthrough')
        except CvBridgeError as e:
            print(e)

        with torch.no_grad():
            self.detect(self.my_img)
    
    def detect(self, my_img):
        img, im0 = self.preprocess_img(my_img)
        # Run inference
        t0 = time.time()
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.nms_thres)
        
        # Apply
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, im0)
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            s = '%g: ' % i

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.classes[int(c)])  # add to string

                # Write results
                for *xyxy, conf, _, cls in reversed(det):
                    label = '%s %.2f' % (self.classes[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)])
                    
        print('Done. (%.3fs)' % (time.time() - t0))
        try:
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(im0, "passthrough"))
        except CvBridgeError as e:
            print(e)
        
        if self.opt.save_video:
            # vid_path = './video.mp4'
            # if isinstance(self.vid_writer, cv2.VideoWriter):
            #     self.vid_writer.release()  # release previous video writer

            # fps = 10
            # h, w, _ = im0.shape
            # self.vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            # self.vid_writer.write(im0)   
            cv2.imwrite('./output/'+str(self.number)+'.png', im0)    
            self.number += 1           
            
        
    
    
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
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/custom.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    # parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    # parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save_video', action='store_true', help='save vidoe')
    opt = parser.parse_args()
    print(opt)

    detector = Detector(opt)
    rospy.spin()
