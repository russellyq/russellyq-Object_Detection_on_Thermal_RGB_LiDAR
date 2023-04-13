import	rospy
from sensor_msgs.msg import CompressedImage, PointCloud2, Image, PointCloud
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pcl2
import numpy as np
import time

class Lidar2Depth:
    def __init__(self):
        self.P_rgb = np.array([[332.232689, 0.000000, 333.058485, 0], 
                                    [0, 332.644823, 240.998586, 0], 
                                    [0, 0, 1, 0]])
        self.LiDAR2rgb = np.array([-0.01807319, -0.99976844,  0.01174591, -0.08543468,
                                    -0.03244301, -0.01115523, -0.99940997,  0.11639403,
                                    0.99931043, -0.01844365, -0.03223318, -0.04211223]).reshape((3,4))
        self.rgb2Radar = np.array([-0.02236591803543918, 0.01373899491374072, 0.9996551471376557, 0.04156851224667894,
                                    -0.9997463777260759, -0.003016115343759606, -0.02232651169926307,  0.06252061356319374,
                                    0.002708378892805941, -0.9999009961676391, 0.01380290382175741, 0.06251000189496365]).reshape((3,4))
        self.Radar2rgb = np.zeros((3, 4))
        self.Radar2rgb[0:3, 0:3] = np.linalg.inv(self.rgb2Radar[0:3, 0:3])
        self.Radar2rgb[0:3, -1] = - np.matmul(self.Radar2rgb[0:3, 0:3], self.rgb2Radar[0:3, -1])
        self.bridge = CvBridge()
        
        rostopic_lists = ['/thermal_cam/thermal_image/compressed',
                          '/rgb_cam/image_raw/compressed',
                          '/livox/lidar/time_sync',
                          '/radar_pcl']
        
        rospy.init_node('ranging2depth', anonymous=True)
        

        self.image_sub = message_filters.Subscriber(rostopic_lists[1], CompressedImage)
        self.lidar_sub = message_filters.Subscriber(rostopic_lists[2], PointCloud2)
        self.radar_sub = message_filters.Subscriber(rostopic_lists[3], PointCloud)
        
        self.ts_img_lidar = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.lidar_sub], 1, 0.1)
        self.ts_img_lidar.registerCallback(self.img_lidar_CallBack)
        
        self.ts_img_radar = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.radar_sub], 1, 0.1)
        self.ts_img_radar.registerCallback(self.img_radar_CallBack)
        
        self.lidar_depth_pub = rospy.Publisher('/lidar_cam/depth_img', Image, queue_size=1)
        self.radar_depth_pub = rospy.Publisher('/radar_cam/depth_img', Image, queue_size=1)


    def img_lidar_CallBack(self, img_msg, lidar_msg):
        t0 = time.time()
        print("Recieve")
        try:
            self.rgb_img = self.bridge.compressed_imgmsg_to_cv2(img_msg, 'passthrough')
            h, w, _ = self.rgb_img.shape
        except CvBridgeError as e:
            print(e)
        points_list = []
        for point in pcl2.read_points(lidar_msg, skip_nans=True, field_names=("x", "y", "z")):
            if point[0] == 0 and point[1] == 0 and point[2] == 0:
                continue
            points_list.append([point[0], point[1], point[2]])
        points_list = np.asarray(points_list)

        points_list = self.lidar_to_camera_3d(points_list, self.LiDAR2rgb)
        points_2d_with_depth, points_2d_xy = self.project_to_img(points_list)

        # draw depth image
        depth_img = np.zeros((h,w)).astype(np.float32)
        depth_img_msg = Image()
        for point, xy in zip(points_2d_with_depth, points_2d_xy):
            u,v,z = point[0], point[1], point[2]
            x, y = xy[0], xy[1]

            if u >= 0 and u < w and v >= 0 and v < h:
                depth_img[int(v),int(u)] = np.float32(z)

        #print(np.max(depth_img))
        depth_img_msg = self.bridge.cv2_to_imgmsg(depth_img, "passthrough")
        depth_img_msg.header = img_msg.header

        self.lidar_depth_pub.publish(depth_img_msg)
        
        rospy.loginfo('Publishing LiDAR Depth Image')
        print('Done: (%.2fs)' % (time.time() - t0))
    
    def img_radar_CallBack(self, img_msg, radar_msg):
        t0 = time.time()
        print("Recieve")
        try:
            self.rgb_img = self.bridge.compressed_imgmsg_to_cv2(img_msg, 'passthrough')
            h, w, _ = self.rgb_img.shape
        except CvBridgeError as e:
            print(e)

        radar_points_list = read_radar_points(radar_msg)

        points_list = self.lidar_to_camera_3d(radar_points_list, self.Radar2rgb)
        points_2d_with_depth, points_2d_xy = self.project_to_img(points_list)

        # draw depth image
        depth_img = np.zeros((h,w)).astype(np.float32)
        depth_img_msg = Image()
        for point, xy in zip(points_2d_with_depth, points_2d_xy):
            u,v,z = point[0], point[1], point[2]
            x, y = xy[0], xy[1]

            if u >= 0 and u < w and v >= 0 and v < h:
                depth_img[int(v),int(u)] = np.float32(z)

        #print(np.max(depth_img))
        depth_img_msg = self.bridge.cv2_to_imgmsg(depth_img, "passthrough")
        depth_img_msg.header = img_msg.header

        self.lidar_depth_pub.publish(depth_img_msg)
        
        rospy.loginfo('Publishing RADAR Depth Image')
        print('Done: (%.2fs)' % (time.time() - t0))

    def lidar_to_camera_3d(self, lidar_pcd, extrinsic):
        # lidar pcd: n * 3
        # return n * 3 (x,y,z)
        n, _ = lidar_pcd.shape
        lidar_pcd = np.c_[lidar_pcd, np.ones((n, 1))]
        camera_pcd = np.matmul(extrinsic, lidar_pcd.T).T
        return camera_pcd
    
    def project_to_img(self, pts_3d):
        # pts_3d: n x 3
        # P: 3 x 4
        # return: n x 2
        z = pts_3d[:, 2]
        n , _ = pts_3d.shape
        pts_3d = np.c_[pts_3d, np.ones((n, 1))]
        camera_uv = np.matmul(self.P_rgb, pts_3d.T).T
        camera_xy = camera_uv[:, 0:2]
        camera_uv = camera_uv[:,0:2] / camera_uv[:, 2:3]
        #print(camera_uv)
        return np.c_[camera_uv[:, 0:2], np.array(z).reshape(n, -1)], camera_xy

def read_radar_points(cloud_msg):
    raw_data = cloud_msg.points
    points = []
    for point in raw_data:
        if point.x == 0 and point.y == 0 and point.z == 0:
            continue
        points.append([point.x, point.y, point.z])
    return np.array(points).reshape((-1, 3))

if __name__ == '__main__':  
    lidar2depth = Lidar2Depth()
    while not rospy.is_shutdown():
        rospy.spin()