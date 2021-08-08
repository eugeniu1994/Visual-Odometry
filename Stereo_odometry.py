import numpy as np
import cv2
import yaml
import g2o
import open3d as o3d
from numbers import Number

class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self, verbose=False):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        super().set_verbose(verbose)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, is_fixed=False):
        v_se2 = g2o.VertexSE3()
        v_se2.set_id(id)
        v_se2.set_estimate(g2o.Isometry3d(pose))
        v_se2.set_fixed(is_fixed)
        super().add_vertex(v_se2)

    def add_edge(self, vertices, measurement=None, information=np.eye(6), robust_kernel=None):
        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(g2o.Isometry3d(measurement))  # relative pose transformation between frames
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

lk_params = dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

class Odometry(object):
    def __init__(self, local_pose_graph=False, local_window=70, display_images=False, ICP_max_iter=70):
        self.start_idx = 1  # 00
        self.N = 2279 - 1  # The number of images
        # if local_pose_graph is False, performe pose graph optimization for all poses, else for a window of poses
        self.local_pose_graph = local_pose_graph
        self.local_window = local_window
        self.display_images = display_images
        self.pose = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]).astype(float)
        self.read_camera_data()  # read camera intrinsic/extrinsic parameters

        self.trajectory = []
        self.trajectory.append(self.pose)
        self.rk = g2o.RobustKernelDCS()
        self.graph_optimizer = PoseGraphOptimization()

        self.threshold = 0.1  # 0.2  # Distance threshold
        self.trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0, 0],
                                      [0.0, 0.0, 0.0,
                                       1.0]])  # Initial transformation matrix, generally provided by coarse registration
        self.ICP_max_iter = ICP_max_iter  # Set the maximum number of iterations for ICP algorithm
        self.width, self.height = 752, 480

    def read_camera_data(self):
        '''read camera intrinsic/extrinsic parameters'''

        def euler_from_matrix(R):
            beta = -np.arcsin(R[2, 0])
            alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
            gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
            return np.array((alpha, beta, gamma))

        # extrinsic-----------------------------------
        self.R = np.array([[0.999966347530033, -0.001422739138722922, 0.008079580483432283],
                           [0.001365741834644127, 0.9999741760894847, 0.007055629199258132],
                           [-0.008089410156878961, -0.007044357138835809, 0.9999424675829176]
                           ])
        angles = euler_from_matrix(self.R)
        print('Rotation angles degree: ', [(180.0 / np.pi) * i for i in angles])
        self.baseline_px = -47.90639384423901
        self.baseline_m = -47.90639384423901 / 435.2046959714599
        self.focal_length = 435.2046959714599  # px
        self.fxypxy = [435.2046959714599, 435.2046959714599, 367.4517211914062, 252.2008514404297]  # fx,fy,px,py
        print('baseline (px) : {},  baseline (m) : {}'.format(self.baseline_px, self.baseline_m))
        self.P1 = np.array([
            [435.2046959714599, 0, 367.4517211914062, 0],
            [0, 435.2046959714599, 252.2008514404297, 0],
            [0, 0, 1, 0]
        ])
        self.P2 = np.array([
            [435.2046959714599, 0, 367.4517211914062, -47.90639384423901],
            [0, 435.2046959714599, 252.2008514404297, 0],
            [0, 0, 1, 0]
        ])

        # intrinsic-------------------------------------
        with open('stereo_data_set/cam1.yaml') as file:
            cam1 = yaml.full_load(file)
            T1 = np.array(cam1['T_BS']['data']).reshape(4, 4)
            i = np.array(cam1['intrinsics'])
            self.D1 = np.array(cam1['distortion_coefficients'])
            self.K1 = np.array([[i[0], 0, i[2]],
                                [0, i[1], i[3]],
                                [0, 0, 1]])
            print('K1 {}'.format(np.shape(self.K1)))
            print(self.K1)
            print('D1 {}'.format(np.shape(self.D1)))
            print(self.D1)

        with open('stereo_data_set/cam2.yaml') as file:
            cam2 = yaml.full_load(file)
            i = np.array(cam2['intrinsics'])
            self.D2 = np.array(cam2['distortion_coefficients'])
            self.K2 = np.array([[i[0], 0, i[2]],
                                [0, i[1], i[3]],
                                [0, 0, 1]])
            print('K2 {}'.format(np.shape(self.K2)))
            print(self.K2)
            print('D2 {}'.format(np.shape(self.D2)))
            print(self.D2)

    # can be used instead of openCV goodFeature to track
    def create_detector(self, orb=True):
        if orb:
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
            search_params = {}
            self.detector = cv2.ORB_create(nfeatures=256)  # nfeatures=256
        else:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=25)
            self.detector = cv2.SIFT_create(nfeatures=128)  # nfeatures = 128
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match_frames(self, img1, img2):
        # compute keypoints and descriptors
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)

        # match keypoints
        matches = self.flann.knnMatch(des1, des2, k=2)
        good_matches_0 = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches_0.append(m.queryIdx)

        des1, des2 = des1[good_matches_0], des2[good_matches_0]
        kp1 = [kp1[m] for m in good_matches_0]
        kp2 = [kp2[m] for m in good_matches_0]
        N = len(good_matches_0)
        px_1 = np.float32([kp1[m].pt for m in range(N)]).reshape(-1, 2)
        px_2 = np.float32([kp2[m].pt for m in range(N)]).reshape(-1, 2)

        return kp1, kp2, des1, des2, px_1, px_2

    def show_features(self, img, px):
        '''display detected features'''
        for p in px:
            cv2.circle(img, tuple(p), 2, (0, 255, 0), 2)
        return img

    def show_images(self, img1, img2, img3, img4, px_1, px_2, px_3, px_4, timp=0, show=False):
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2RGB)
        img4 = cv2.cvtColor(img4, cv2.COLOR_GRAY2RGB)

        img1 = self.show_features(img1, px_1)
        img2 = self.show_features(img2, px_2)
        img3 = self.show_features(img3, px_3)
        img4 = self.show_features(img4, px_4)

        s1 = np.hstack((img1, img2))
        s2 = np.hstack((img3, img4))
        s = np.vstack((s1, s2))
        self.s = cv2.resize(s, None, fx=.7, fy=.7)
        if show:
            cv2.imshow('Images', self.s)
            # cv2.waitKey(timp)
        else:
            return s

    def match_frames_LK(self, img1, img2, img3, img4, verbose=False):
        '''detect and match the features across 4 images'''
        px_1 = np.asarray(cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)).squeeze()
        px_2, _, _ = cv2.calcOpticalFlowPyrLK(prevImg=img1, nextImg=img2, prevPts=px_1, nextPts=None, **lk_params)
        px_3, _, _ = cv2.calcOpticalFlowPyrLK(prevImg=img1, nextImg=img3, prevPts=px_1, nextPts=None, **lk_params)
        px_4, _, _ = cv2.calcOpticalFlowPyrLK(prevImg=img1, nextImg=img4, prevPts=px_1, nextPts=None, **lk_params)
        if verbose:
            print('px_1->{}, px_2->{}, px_3->{}, px_4->{}'.format(np.shape(px_1), np.shape(px_2), np.shape(px_3),
                                                                  np.shape(px_4)))

        return px_1, px_2, px_3, px_4

    def compute_3D(self, px1, px2):
        '''estimate the 3D pointcloud based on sparse disparity '''
        disparity = np.sum(np.sqrt((px1 - px2) ** 2), axis=1)
        baseline = self.baseline_m
        depth = (baseline * self.focal_length / disparity)
        _3DPoints = []
        for i, pixel in enumerate(px1):
            u, v = pixel.ravel()
            u, v = int(u), int(v)
            Z = depth[i]
            pt = np.array([u, v, Z])

            pt[0] = pt[2] * (pt[0] - self.fxypxy[2]) / self.fxypxy[0]
            pt[1] = pt[2] * (pt[1] - self.fxypxy[3]) / self.fxypxy[1]

            _3DPoints.append(np.array(pt).squeeze())
        return np.array(_3DPoints).squeeze()

    def kill(self):
        print('----Kill the node----')
        cv2.destroyAllWindows()

    def plot_ground_truth(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from pytransform3d.rotations import plot_basis

        fig = plt.figure(figsize=(8, 8))
        fig.suptitle('ground truth point clouds', fontsize=12)
        ax = plt.axes(projection='3d')

        ax.clear()
        plot_basis(ax=ax, R=None, p=np.zeros(3), s=.3)
        for i, R in enumerate(self.Rs):
            t = self.ts[i]
            ax.scatter(t[0], t[1], t[2], marker='o', c='b', alpha=1, s=10)  # shows translation
            plot_basis(ax=ax, R=self.R, p=t, s=.5)  # shows rotation
        ax.set_xlabel('X(m)')
        ax.set_ylabel('Y(m)')
        ax.set_zlabel('Z(m)')
        ax.grid()

        fig.canvas.draw_idle()
        plt.pause(0.001)
        plt.show()

    def get_color(self, img, pt):
        x = int(np.clip(pt[0], 0, self.width - 1))
        y = int(np.clip(pt[1], 0, self.height - 1))
        color = img[y, x]
        if isinstance(color, Number):
            color = np.array([color, color, color])
        return color[::-1] / 255.

    def solve_pnp_ransac(self, pts3d, pts):
        val, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.array(pts3d), np.array(pts),
            self.K1, None, None, None,
            False, 100, 2.0, 0.9, None)

        print('inliers -> {}/{}'.format(len(inliers), len(pts)))
        if inliers is None or len(inliers) < 5:
            print('Not enought inliers')
            return None, None, None, None
        R = cv2.Rodrigues(rvec)[0]
        T = g2o.Isometry3d(R, tvec).matrix()
        return T, inliers.ravel(), R, tvec

    def do_job(self):
        if self.local_pose_graph is False:
            id = 0  # first node of the posegraph
            self.graph_optimizer.add_vertex(id, self.pose, is_fixed=True)
        from viewer import Viewer
        viewobj = Viewer()
        for i in range(self.start_idx, self.N):
            print('i -> {}'.format(i))
            # Capture images
            imgPath = 'stereo_data_set/images/left_{}.png'.format(i - 1)
            I_l1 = cv2.imread(imgPath, 0)
            imgPath = 'stereo_data_set/images/right_{}.png'.format(i - 1)
            I_r1 = cv2.imread(imgPath, 0)
            imgPath = 'stereo_data_set/images/left_{}.png'.format(i)
            I_l2 = cv2.imread(imgPath, 0)
            imgPath = 'stereo_data_set/images/right_{}.png'.format(i)
            I_r2 = cv2.imread(imgPath, 0)

            # undistort images-----------------------------------------
            I_l1 = cv2.undistort(I_l1, self.K1, self.D1, None, self.K1)
            I_l2 = cv2.undistort(I_l2, self.K1, self.D1, None, self.K1)
            I_r1 = cv2.undistort(I_r1, self.K2, self.D2, None, self.K2)
            I_r2 = cv2.undistort(I_r2, self.K2, self.D2, None, self.K2)

            # compute the features and track them across all 4 image
            px_1, px_2, px_3, px_4 = self.match_frames_LK(img1=I_l1, img2=I_r1, img3=I_l2, img4=I_r2, verbose=False)

            if self.display_images:
                self.show_images(I_l1, I_r1, I_l2, I_r2, px_1, px_2, px_3, px_4, timp=1)

            # compute 3D point cloud based on disparity
            W1 = self.compute_3D(px1=px_1, px2=px_2)
            W2 = self.compute_3D(px1=px_3, px2=px_4)

            # Front-end -> ICP compute initial T between pointclouds
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(W1)
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(W2)
            self.threshold = 0.2  # 0.2  # Distance threshold
            icp_p2p = o3d.pipelines.registration.registration_icp(source, target, self.threshold, self.trans_init,
                                                                  o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                                  # Execute point-to-point ICP algorithm
                                                                  o3d.pipelines.registration.ICPConvergenceCriteria(
                                                                      max_iteration=self.ICP_max_iter))
            T = icp_p2p.transformation
            if len(W2) > 6 and False:
                print('Here -> 3D:{}, 2D:{}'.format(np.shape(W2), np.shape(px_1)))
                self.R, self.t = cv2.Rodrigues(T[:3, :3])[0], np.array(T[:3, -1], dtype=np.float32).reshape(-1, 3)
                print('rvec: {}'.format(self.R))
                print('tvec: {}'.format(self.t))
                success, rvec, tvec = cv2.solvePnP(
                    objectPoints=np.array(W2),
                    imagePoints=np.array(px_1),
                    cameraMatrix=self.K1,
                    distCoeffs=self.D1,
                    rvec=self.R, tvec=self.t
                )

                print('success -> {}'.format(success))
                print('tvec: {}'.format(tvec))
                R = cv2.Rodrigues(rvec)[0]
                T = g2o.Isometry3d(R, tvec).matrix()

            colour = []
            for p in px_1:
                colour.append(self.get_color(img=I_l1, pt=p))
            self.pose = T @ self.pose          #update the current pose
            #self.pose = np.dot(T, self.pose)

            if self.local_pose_graph is False:
                # Back-end -> posegraph optimization
                id += 1
                self.graph_optimizer.add_vertex(id=id, pose=self.pose)
                self.graph_optimizer.add_edge(vertices=[id - 1, id], measurement=T, robust_kernel=self.rk)
                self.graph_optimizer.optimize()

                self.nodes_optimized = [i.estimate().matrix() for i in self.graph_optimizer.vertices().values()]
                self.pose = np.asarray(self.nodes_optimized[0]).squeeze()
            else:  # performe pose-graph optimization for a small subset of poses
                if len(self.trajectory) < self.local_window + 2:
                    continue
                self.graph_optimizer = PoseGraphOptimization()

                local_poses = self.trajectory[-self.local_window + 1:]
                self.graph_optimizer.add_vertex(0, local_poses[0], is_fixed=True)
                for j in range(1, len(local_poses)):
                    T = self.Ts[j - 1]
                    self.graph_optimizer.add_vertex(id=j, pose=local_poses[j])
                    self.graph_optimizer.add_edge(vertices=[j - 1, j], measurement=T, robust_kernel=self.rk)
                    self.graph_optimizer.optimize()

                selfnodes_optimized = [np.asarray(i.estimate().matrix()) for i in
                                       self.graph_optimizer.vertices().values()]
                self.pose = np.asarray(selfnodes_optimized)[::-1]

            #T = self.pose
            T = g2o.Isometry3d(self.pose).inverse().matrix()
            self.R, self.t = T[:3, :3], np.array(T[:3, -1]).reshape(-1, 3)
            colour = np.array(colour)
            self.R, self.t = T[:3, :3], np.array(T[:3, -1])
            cloud = W1
            cloud = cloud@np.eye(3) + T[:3, -1]
            viewobj.update_pose(g2o.Isometry3d( self.R, self.t), cloud=cloud, colour=colour)
            viewobj.update_image(self.s)
            print()

        print('-----Done-----')

if __name__ == '__main__':
    local_pose_graph = False  # perform local pose prapg optimization if its True
    local_window = 50  # the number of poses used for local optimization
    display_images = True  # shows the images with the detected fetures
    ICP_max_iter = 100  # the number of iteration for ICP in front-end to estimate the initial transformation

    node = Odometry(local_pose_graph=local_pose_graph, local_window=local_window, display_images=display_images,
                    ICP_max_iter=ICP_max_iter)
    node.do_job()
    node.plot_ground_truth()
    node.kill()
