import numpy as np
import cv2
import g2o
from collections import defaultdict
from queue import Queue
from threading import Thread
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from viewer import Viewer

class Frame(object):
    def __init__(self, image, cam=None, pose=None):
        self.image = image
        self.cam = cam
        self.pose = pose
        self.kp = None
        self.desc = None

    def extract(self, detector, extractor):
        self.kp = detector.detect(self.image)
        self.kp, self.desc = extractor.compute(
            self.image, self.kp)

class StereoFrame(object):
    def __init__(self, left_img, right_img, cam=None, pose=None):
        self.image = left_img
        self.cam = None
        self.pose = pose
        self.left = Frame(image = left_img)
        self.right = Frame(image = right_img)
        if cam is not None:
            self.set_camera(cam)

        self.triangulated_keypoints = []
        self.triangulated_descriptors = []
        self.triangulated_points = []

    def set_camera(self, cam):
        self.cam = cam
        self.left.cam = cam.left_cam
        self.right.cam = cam.right_cam

    def extract(self, detector, extractor):
        t2 = Thread(target=self.right.extract, args=(detector, extractor))
        t2.start()
        self.left.extract(detector, extractor)
        t2.join()

        #self.left.extract(detector, extractor)
        #self.right.extract(detector, extractor)

    def set_triangulated(self, i, point):
        self.triangulated_keypoints.append(self.left.kp[i])
        self.triangulated_descriptors.append(self.left.desc[i])
        self.triangulated_points.append(point)

    @property
    def keypoints(self):
        return self.left.kp

    @property
    def descriptors(self):
        return self.left.desc

class Camera(object):
    def __init__(self,width, height,
                 intrinsic_matrix,undistort_rectify=False,
                 extrinsic_matrix=None,distortion_coeffs=None,
                 rectification_matrix=None,projection_matrix=None):
        self.width = width
        self.height = height
        self.intrinsic_matrix = intrinsic_matrix
        self.extrinsic_matrix = extrinsic_matrix
        self.distortion_coeffs = distortion_coeffs
        self.rectification_matrix = rectification_matrix
        self.projection_matrix = projection_matrix
        self.undistort_rectify = undistort_rectify
        self.fx = intrinsic_matrix[0, 0]
        self.fy = intrinsic_matrix[1, 1]
        self.cx = intrinsic_matrix[0, 2]
        self.cy = intrinsic_matrix[1, 2]

        if undistort_rectify:
            self.remap = cv2.initUndistortRectifyMap(
                cameraMatrix=self.intrinsic_matrix,
                distCoeffs=self.distortion_coeffs,
                R=self.rectification_matrix,
                newCameraMatrix=self.projection_matrix,
                size=(width, height),
                m1type=cv2.CV_8U)
        else:
            self.remap = None

    def rectify(self, img):
        if self.remap is None:
            return img
        else:
            return cv2.remap(img, *self.remap, cv2.INTER_LINEAR)

class StereoCamera(object):
    def __init__(self, left_cam, right_cam):
        self.left_cam = left_cam
        self.right_cam = right_cam

        self.width = left_cam.width
        self.height = left_cam.height
        self.intrinsic_matrix = left_cam.intrinsic_matrix
        self.extrinsic_matrix = left_cam.extrinsic_matrix
        self.fx = left_cam.fx
        self.fy = left_cam.fy
        self.cx = left_cam.cx
        self.cy = left_cam.cy
        self.baseline = abs(right_cam.projection_matrix[0, 3] /
            right_cam.projection_matrix[0, 0])
        self.focal_baseline = self.fx * self.baseline

class VOConfig(object):
    def __init__(self, feature='GFTT'):
        self.epipolar_range = 1.0  # pixel
        self.max_depth = 20  # meter
        self.min_depth = 0.01  # meter
        if feature == 'GFTT':
            self.detector = cv2.GFTTDetector_create(
                maxCorners=500, qualityLevel=0.01, minDistance=9, blockSize=9)
            self.extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)
        elif feature == 'ORB':
            self.detector = cv2.ORB_create(
                nfeatures=1000, scaleFactor=1.2, nlevels=8,
                edgeThreshold=31, patchSize=31)
            self.extractor = self.detector
        else:
            raise NotImplementedError

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.dist_std_scale = 0.4

        self.disparity_matches = 20# 40
        self.min_good_matches = 20# 40
        self.matching_distance = 25
        self.min_inliers = 20# 40
        self.restart_tracking = 3

        self.max_update_time = 0.45

        self.cell_size = 15

class ImageReader(object):
    def __init__(self, ids, cam):
        self.ids = ids
        self.cam = cam
        self.cache = dict()
        self.idx = 0
        self.ahead = 10  # 10 images ahead of current index
        self.wait = 1.5  # waiting time

        self.preload_thread = Thread(target=self.preload)
        self.thread_started = False

    def read(self, path):
        img = cv2.imread(path, -1)
        return self.cam.rectify(img)

    def preload(self):
        idx = self.idx
        t = float('inf')
        while True:
            if time.time() - t > self.wait:
                return
            if self.idx == idx:
                time.sleep(1e-2)
                continue

            for i in range(self.idx, self.idx + self.ahead):
                if i not in self.cache and i < len(self.ids):
                    self.cache[i] = self.read(self.ids[i])
            if self.idx + self.ahead > len(self.ids):
                return
            idx = self.idx
            t = time.time()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.idx = idx
        if not self.thread_started:
            self.thread_started = True
            self.preload_thread.start()

        if idx in self.cache:
            img = self.cache[idx]
            del self.cache[idx]
        else:
            img = self.read(self.ids[idx])
        return img

    def __iter__(self):
        for i, _ in enumerate(self.ids):
            yield self[i]

    @property
    def starttime(self):
        return self.ids[0]

class Stereo(object):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __iter__(self):
        for l, r in zip(self.left, self.right):
            yield (l, r)

    def __len__(self):
        return len(self.left)

    @property
    def starttime(self):
        return self.left.starttime

class EuRoCDataset(object):  # Stereo + IMU
    def __init__(self, path, rectify=True):
        self.left_cam = Camera(width=752, height=480,
            intrinsic_matrix=np.array([
                [458.654, 0.000000, 367.215],
                [0.000000, 457.296, 248.375],
                [0.000000, 0.000000, 1.000000]]),
            undistort_rectify=rectify,
            distortion_coeffs=np.array(
                [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.000000]),
            rectification_matrix=np.array([
                [0.999966347530033, -0.001422739138722922, 0.008079580483432283],
                [0.001365741834644127, 0.9999741760894847, 0.007055629199258132],
                [-0.008089410156878961, -0.007044357138835809, 0.9999424675829176]]),
            projection_matrix=np.array([
                [435.2046959714599, 0, 367.4517211914062, 0],
                [0, 435.2046959714599, 252.2008514404297, 0],
                [0., 0, 1, 0]]),
            extrinsic_matrix=np.array([
                [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                [0.0, 0.0, 0.0, 1.0]])
        )
        self.right_cam = Camera(width=752, height=480,
            intrinsic_matrix=np.array([
                [457.587, 0.000000, 379.999],
                [0.000000, 456.134, 255.238],
                [0.000000, 0.000000, 1.000000]]),
            undistort_rectify=rectify,
            distortion_coeffs=np.array(
                [-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0]),
            rectification_matrix=np.array([
                [0.9999633526194376, -0.003625811871560086, 0.007755443660172947],
                [0.003680398547259526, 0.9999684752771629, -0.007035845251224894],
                [-0.007729688520722713, 0.007064130529506649, 0.999945173484644]]),
            projection_matrix=np.array([
                [435.2046959714599, 0, 367.4517211914062, -47.90639384423901],
                [0, 435.2046959714599, 252.2008514404297, 0],
                [0, 0, 1, 0]]),
            extrinsic_matrix=np.array([
                [0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556],
                [0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
                [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
                [0.0, 0.0, 0.0, 1.0]])
        )
        path = os.path.expanduser(path)
        self.left = ImageReader(ids=self.list_imgs(path),cam=self.left_cam)
        self.right = ImageReader(ids=self.list_imgs(path),cam=self.right_cam)
        assert len(self.left) == len(self.right)

        self.stereo = Stereo(self.left, self.right)
        self.cam = StereoCamera(self.left_cam, self.right_cam)

    @property
    def starttime(self):
        return self.stereo.starttime

    def list_imgs(self, dir, left=True):
        #xs = [_ for _ in os.listdir(dir) if _.endswith('.png')]
        xs = [_ for _ in os.listdir(dir) if _.__contains__('left' if left else "right")]
        split = 5 if left else 6
        xs = sorted(xs, key=lambda x: float(x[split:-4]))
        return [os.path.join(dir, _) for _ in xs]

class StereoVO(object):
    def __init__(self, cam, config, show=True):
        self.cam = cam
        self.config = config
        self.show = show

        self.detector = config.detector
        self.extractor = config.extractor
        self.matcher = config.matcher

        self.keyframe = None
        self.candidates = []

        self.pose = np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]).astype(float)

    def track(self, frame, wait=1):
        frame.set_camera(self.cam)
        frame.extract(self.detector, self.extractor)
        try:
            self.triangulate(frame)
        except Exception as err:
            print('return None self.triangulate -> {}'.format(err))
            return None

        if self.keyframe is None:
            self.keyframe = frame
            return None

        pts3, pts2, matches = self.match(self.keyframe, frame)
        if len(matches) < self.config.min_good_matches:
            self.restart_tracking(frame)
            return None
        # transform points in keyfram to current frame
        T, inliers, R, t = solve_pnp_ransac(pts3, pts2, frame.cam.intrinsic_matrix)
        if self.show or True:
            #print('inliers -> {}'.format(inliers))
            img3 = cv2.drawMatches(self.keyframe.image,self.keyframe.triangulated_keypoints,
                frame.image,frame.keypoints,[matches[i] for i in inliers],None,flags=2)
            cv2.imshow('transform points in keyfram to current frame', img3)
            cv2.waitKey(wait)

        if T is None or len(inliers) < self.config.min_inliers:
            #print(' -> len(inliers)={}'.format(len(inliers)))
            self.restart_tracking(frame)
            return None

        self.candidates.append(frame)
        #T_new = transformMatrix(R, t)
        T_new = T
        #self.pose = T_new @ self.pose  # update the current pose
        self.pose = np.dot(self.pose, T_new)

        return T, R, t

    def restart_tracking(self, frame):
        #print('restart_tracking')
        if len(self.candidates) > 4:
            self.keyframe = self.candidates[-2]
        if len(self.candidates) > 0:
            self.keyframe = self.candidates[-1]
        else:
            self.keyframe = frame
        self.candidates.clear()

    def match(self, query_frame, match_frame):
        # TODO: use predicted pose from IMU or motion model
        matches = self.matcher.match(np.array(query_frame.triangulated_descriptors),np.array(match_frame.descriptors))
        distances = defaultdict(lambda: float('inf'))
        good = dict()
        for m in matches:
            pt = query_frame.triangulated_keypoints[m.queryIdx].pt
            id = (int(pt[0] / self.config.cell_size),
                  int(pt[1] / self.config.cell_size))
            if m.distance > min(self.config.matching_distance, distances[id]):
                continue
            good[id] = m
            distances[id] = m.distance
        good = list(good.values())
        pts3,pts2 = [],[]
        for m in good:
            pts3.append(query_frame.triangulated_points[m.queryIdx])
            pts2.append(match_frame.keypoints[m.trainIdx].pt)
        return pts3, pts2, good

    def triangulate(self, frame):
        matches = self.matcher.match(frame.left.desc, frame.right.desc)
        assert len(matches) > self.config.disparity_matches
        good = []
        for i, m in enumerate(matches):
            query_pt = frame.left.kp[m.queryIdx].pt
            match_pt = frame.right.kp[m.trainIdx].pt
            dx = abs(query_pt[0] - match_pt[0])
            dy = abs(query_pt[1] - match_pt[1])
            if dx == 0:
                continue
            depth = frame.cam.focal_baseline / dx
            #if (dy <= self.config.epipolar_range and self.config.min_depth <= depth <= self.config.max_depth):
            if (self.config.min_depth <= depth <= self.config.max_depth):
                good.append(m)

                point = np.zeros(3)
                point[2] = depth
                point[0] = (query_pt[0] - frame.cam.cx) * depth / frame.cam.fx
                point[1] = (query_pt[1] - frame.cam.cy) * depth / frame.cam.fy
                frame.set_triangulated(m.queryIdx, point)

        assert len(good) > self.config.disparity_matches

def solve_pnp_ransac(pts3d, pts, intrinsic_matrix):
    print('pts3d')
    print(pts3d)
    print('pts')
    print(pts)
    print('intrinsic_matrix')
    print(intrinsic_matrix)
    val, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.array(pts3d), np.array(pts),
        intrinsic_matrix, None, #None, None,
        #False, 50, 2.0, 0.9, None
    )
    print('val -> {}'.format(val))
    #print('rvec -> {}'.format(rvec))
    if inliers is None or len(inliers) < 5:
        return None, None
    R = cv2.Rodrigues(rvec)[0]
    tvec = np.zeros_like(tvec)
    T = g2o.Isometry3d(R, tvec).matrix()
    return T, inliers.ravel(), R, tvec

def transformMatrix(rvec, tvec):
    R, t = np.matrix(rvec), np.matrix(tvec)
    Rt = np.hstack((R, t))
    T = np.vstack((Rt, np.matrix([0, 0, 0, 1])))
    return T

if __name__ == '__main__':
    viewobj = Viewer()
    config = VOConfig()
    dataset = EuRoCDataset(path='stereo_data_set/images')
    vo = StereoVO(dataset.cam, VOConfig(), True)

    dataset = iter(dataset.stereo)
    while True:
        data = next(dataset)
        if data is None:
            break
        (left_img, right_img) = data
        frame = StereoFrame(left_img, right_img)
        vo.track(frame)

        viewobj.update_pose(g2o.Isometry3d(vo.pose))
        viewobj.update_image(left_img)
        #break
        #print(vo.pose)



