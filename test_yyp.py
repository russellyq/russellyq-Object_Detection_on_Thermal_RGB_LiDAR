import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt
import sys
import json
from yolov3 import TrtYOLOv3

###################################### DATA DIRECTORY ######################################
## TESTING DATA
#dataDir = "/home/yyp/dissertation/visible_ir_demo/visible_ir_demo/data_label"
#rgb_dir = "crop_LR_visible"
#ir_dir = "cropinfrared"

## ALL DATA
dataDir = "/home/yyp/dissertation/visible_ir_demo/visible_ir_demo/data"
rgb_dir = "crop_LR_visible"
ir_dir = "cropinfrared"



# Constants
ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

conf_th = 0.3

class ImagePreprocess():
    def __init__(self,rootdir,img_dir_list) -> None:
        # initialize
        self.rootdir_ = rootdir
        self.rgb_dir_ = img_dir_list[0]
        self.ir_dir_ = img_dir_list[1]
        self.imgname_list_len = self.load_dir(rootdir)
    
    def load_dir(self,rootdir):
        # load images in directory
        self.imgname_list = []
        path = os.path.join(rootdir,self.rgb_dir_)
        subfile = os.listdir(path)
        for j in subfile:
            file_name, file_extension = os.path.splitext(j)
            if (file_extension == ".jpg"):
                self.imgname_list.append(j)
        return len(self.imgname_list)

    def read_image(self,img_name):
        # read image file
        rgb_path = os.path.join(self.rootdir_,self.rgb_dir_,img_name)
        ir_path = os.path.join(self.rootdir_,self.ir_dir_,img_name)

        rgb_img = cv2.imread(rgb_path,cv2.IMREAD_COLOR) # read visible images as rgb
        ir_img = cv2.imread(ir_path,cv2.IMREAD_GRAYSCALE) # read infrared images as grayscale
        
        return rgb_img,ir_img

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

def draw_boxed_text(img, text, topleft, color):
    """Draw a transluent boxed text in white, overlayed on top of a
    colored patch surrounded by a black border. FONT, TEXT_SCALE,
    TEXT_THICKNESS and ALPHA values are constants (fixed) as defined
    on top.

    # Arguments
      img: the input image as a numpy array.
      text: the text to be drawn.
      topleft: XY coordinate of the topleft corner of the boxed text.
      color: color of the patch, i.e. background of the text.

    # Output
      img: note the original image is modified inplace.
    """
    assert img.dtype == np.uint8
    img_h, img_w = img.shape[:2] # img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 3
    size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2
    # the patch is used to draw boxed text
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    cv2.putText(patch, text, (margin+1, h-margin-2), FONT, TEXT_SCALE,
                WHITE, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
    cv2.rectangle(patch, (0, 0), (w-1, h-1), BLACK, thickness=1)
    w = min(w, img_w - topleft[0])  # clip overlay at image boundary
    h = min(h, img_h - topleft[1])
    # Overlay the boxed text onto region of interest (roi) in img
    roi = img[topleft[1]:topleft[1]+h, topleft[0]:topleft[0]+w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img

def draw_bboxes(img, boxes, confs, clss):
        """Draw detected bounding boxes on the original image."""
        if len(img.shape)==2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if not len(boxes):
            return img
        for bb, cf, cl in zip(boxes, confs, clss):
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), 1, 2)
            txt_loc = (max(x_min+2, 0), max(y_min+2, 0))
            cls_name = cl
            txt = '{} {:.2f}'.format(cls_name, cf)
            img = draw_boxed_text(img, txt, txt_loc, 1)
        return img


def nms(boxes,box_confidences,classnames):

    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
    confidence scores and return an array with the indexes of the bounding boxes we want to
    keep (and display later).
    Keyword arguments:
    boxes -- a NumPy array containing N bounding-box coordinates that survived filtering,
    with shape (N,4); 4 for x,y,height,width coordinates of the boxes
    box_confidences -- a Numpy array containing the corresponding confidences with shape N
    """
    result_boxes, result_classnames, result_scores = list(), list(), list()
    x_coord = boxes[:, 0]
    y_coord = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)

        # Compute the Intersection over Union (IoU) score:
        iou = intersection / union

        # The goal of the NMS algorithm is to reduce the number of adjacent bounding-box
        # candidates to a minimum. In this step, we keep only those elements whose overlap
        # with the current bounding box is lower than the threshold:
        indexes = np.where(iou <= 0.8)[0]
        #indexes = np.where(iou <= self.nms_threshold)[0]
        ordered = ordered[indexes + 1]

    keep = np.array(keep)
    
    result_boxes.append(boxes[keep])
    result_classnames.append(classnames[keep])
    result_scores.append(box_confidences[keep])

    boxes = np.concatenate(result_boxes)
    categories = np.concatenate(result_classnames)
    confidences = np.concatenate(result_scores)

    return boxes, categories, confidences

def rgb_ir_fusion_comparsion(model,rgb_img,ir_img,fusion_type):
    ##using nms for fusion
    res_img_rgb=rgb_img
    res_img_ir=ir_img
    ##res_img_rgb = cv2.addWeighted(cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY),1,ir_img,0,0) # rgb
    ##res_img_ir = cv2.addWeighted(cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY),0,ir_img,1,0) # ir

    ret_img_rgb, boxes_rgb, scores_rgb, classes_rgb, classesname_rgb = model.detect(res_img_rgb, conf_th=conf_th, draw=True)
    ret_img_ir, boxes_ir, scores_ir, classes_ir, classesname_ir = model.detect(res_img_ir, conf_th=conf_th,draw=True)

    boxes_fusion=np.vstack((boxes_rgb, boxes_ir))
    scores_fusion=np.append(scores_rgb, scores_ir)
    classesname_fusion=np.append(classesname_rgb, classesname_ir)
    result_boxes,result_classes,result_scores=[],[],[]
    #print(boxes_fusion,type(boxes_fusion))
    #print(scores_fusion,type(scores_fusion))
    #print(classesname_fusion,type(classesname_fusion))
    if classesname_fusion.size:
        result_boxes,result_classes,result_scores=nms(boxes_fusion,scores_fusion,classesname_fusion)
	
	#save result 
        sample_num = result_classes.size
        for i in range(sample_num):
           out_file = open("../yolov3-onnx/result/{}.txt".format(result_classes[i]), 'a')
           #out_file.write(img_name)
           out_file.write(os.path.splitext(img_name)[0])
           out_file.write(' ')
           out_file.write(str(result_scores[i]))
           out_file.write(' ')
           out_file.write(" ".join([str(i) for i in result_boxes[i]])+"\n")
           out_file.close()

        print("--------Fusion--------")
        print("classname = ", result_classes)
        print("scores    = ", result_scores)

    print("#############################")
    print("----------RGB-----------")
    print("classname = ", classesname_rgb)
    print("scores    = ", scores_rgb)
    print("--------INFRARED--------")
    print("classname = ", classesname_ir)
    print("scores    = ", scores_ir)

    ## plot
    fig.suptitle(pltImgTitle)

    ax[0].imshow(cv2.cvtColor(rgb_img,cv2.COLOR_BGR2RGB))
    ax[0].set_title("RGB")
    ax[0].axis("off")

    ax[2].imshow(draw_bboxes(ir_img, result_boxes, result_scores, result_classes))
    ax[2].set_title("Fusion")
    ax[2].axis("off")

    ax[1].imshow(ret_img_ir,cmap="gray")
    ax[1].set_title("Infrared")
    ax[1].axis("off")

    #ax[1,1].imshow(ret_img_rgb,cmap="gray")
    #ax[1,1].set_title("More RGB")
    #ax[1,1].axis("off")

    plt.draw()
    return 

## plot initialization
fig, ax = plt.subplots(1,3)
closed = False
pltImgTitle = ""

def handle_close(evt):
    global closed
    closed = True

def waitforbuttonpress():
    while plt.waitforbuttonpress(0.2) is None:
        if closed:
            return False
    return True



if __name__ == "__main__":
    if sys.version_info[0] < 3:
        raise Exception('yolov3-onnx is only compatible with python3...')

    imageProcessor = ImagePreprocess(dataDir,[rgb_dir,ir_dir])
    day_night_th = 0.6 # empirical value
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_file', default='model_data/images/1.jpg', help='test image file')
    args = parser.parse_args()
    img_file = args.img_file
    """

    model_path = "model_data"
    model_type = "yolov3-288"
    model = TrtYOLOv3(model_path,model_type)

    imagelist = imageProcessor.imgname_list
    for img_name in imagelist:
        rgb_img, ir_img = imageProcessor.read_image(img_name)
        rgb_mean, rgb_std, whiten_rgb_img = imageProcessor.whiten(rgb_img)
        ir_mean, ir_std, whiten_ir_img = imageProcessor.whiten(ir_img)

        fusion_type = 0
        if (rgb_mean >= day_night_th):
            # day time, mainly rgb
            fusion_type = 0
        else:
            # night time, more thermal
            fusion_type = 1
        
        rgb_ir_fusion_comparsion(model,rgb_img,ir_img,fusion_type)
 

        ## save result
        #plt.savefig(os.path.join("result",str(img_name)),dpi=600)

        if not waitforbuttonpress():
            break

