import os
import cv2


vid_path = './video.mp4'
fps = int(25)
h, w = 512, 640
vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

for i in range(1584):
    im = cv2.imread('./output/'+str(i)+'.png')
    vid_writer.write(im)
vid_writer.release()
        