import numpy as np
import OpenGL.GL as gl
import sys
sys.path.append("/home/eugeniu/pangolin")
import pangolin
import cv2
from multiprocessing import Queue, Process

class Viewer(object):
    def __init__(self):
        self.image_queue = Queue()
        self.pose_queue = Queue()
        self.map_queue = Queue()
        self.colour_queue = Queue()

        self.q_points = Queue()

        self.view_thread = Process(target=self.view)
        self.view_thread.start()

    def update_pose(self, pose, cloud=None, colour = None):
        if pose is None:
            return
        self.pose_queue.put(pose.matrix())
        #print(pose.matrix())
        if cloud is not None:
            self.map_queue.put((cloud.tolist(), 1))
        if colour is not None:
            self.colour_queue.put(colour.tolist())

    def update_image(self, image):
        if image is None:
            return
        elif image.ndim == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=2)
        self.image_queue.put(image)

    def view(self):
        pangolin.CreateWindowAndBind('RGB-D SLAM', 1024, 768)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        panel = pangolin.CreatePanel('menu')
        panel.SetBounds(0.5, 1.0, 0.0, 175 / 1024.)

        # checkbox
        m_follow_camera = pangolin.VarBool('menu.Follow Camera', value=True, toggle=True)

        viewpoint_x = 0
        viewpoint_y = -7
        viewpoint_z = -18
        viewpoint_f = 1000

        proj = pangolin.ProjectionMatrix(
            1024, 768, viewpoint_f, viewpoint_f, 512, 389, 0.1, 300)
        look_view = pangolin.ModelViewLookAt(
            viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, 0, -1, 0)

        # Camera Render Object (for view / scene browsing)
        scam = pangolin.OpenGlRenderState(proj, look_view)

        # Add named OpenGL viewport to window and provide 3D Handler
        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(0.0, 1.0, 175 / 1024., 1.0, -1024 / 768.)
        dcam.SetHandler(pangolin.Handler3D(scam))

        # image
        width, height = 376, 240
        width, height = 376*2, 240
        dimg = pangolin.Display('image')
        dimg.SetBounds(0, height / 768., 0.0, width / 1024., 1024 / 768.)
        dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

        texture = pangolin.GlTexture(width, height, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image = np.ones((height, width, 3), 'uint8')

        # axis
        axis = pangolin.Renderable()
        axis.Add(pangolin.Axis())

        trajectory = DynamicArray()
        camera, image = None, None
        mappoints = DynamicArray(shape=(3,))
        pts_color = DynamicArray(shape=(3,))
        pose = pangolin.OpenGlMatrix()  # identity matrix
        following = True

        while not pangolin.ShouldQuit():
            if not self.pose_queue.empty():
                while not self.pose_queue.empty():
                    poses = self.pose_queue.get()
                trajectory.append(poses[:3, 3])
                camera = poses
                pose.m = poses

            if not self.image_queue.empty():
                while not self.image_queue.empty():
                    img = self.image_queue.get()
                img = img[::-1, :, ::-1]
                img = cv2.resize(img, (width, height))
                image = img.copy()

            # Show mappoints
            if not self.map_queue.empty():
                pts, code = self.map_queue.get()
                if code == 1:  # append new points extend mappoints->314, pts->(229, 3), type=<class 'list'>
                    mappoints.extend(pts)
            if not self.colour_queue.empty():
                pts_color.extend(self.colour_queue.get())

            if camera is not  None:
                follow = m_follow_camera.Get()
                if follow and following:
                    scam.Follow(pose, True)
                elif follow and not following:
                    scam.SetModelViewMatrix(look_view)
                    scam.Follow(pose, True)
                    following = True
                elif not follow and following:
                    following = False

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            dcam.Activate(scam)
            # draw axis
            axis.Render()
            # draw current camera
            if camera is not None:
                gl.glLineWidth(1)
                gl.glColor3f(0.0, 0.0, 1.0)
                pangolin.DrawCameras(np.array([camera]), 0.3)

            # show trajectory
            if len(trajectory) > 0:
                gl.glPointSize(8)
                gl.glColor3f(0.0, 1.0, 0.0)
                pangolin.DrawPoints(trajectory.array())

                # show map
            if len(mappoints) > 0 and len(pts_color) > 0:
                gl.glPointSize(2)
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawPoints(mappoints.array(), pts_color.array())
            elif len(mappoints) > 0:
                gl.glPointSize(1)
                gl.glColor3f(0.0, 0.0, 1.0)
                pangolin.DrawPoints(mappoints.array())

            # show image
            if image is not None:
                texture.Upload(image, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                dimg.Activate()
                gl.glColor3f(1.0, 1.0, 1.0)
                texture.RenderToViewport()

            pangolin.FinishFrame()

    def stop(self):
        self.view_thread.join()
        qtype = type(Queue())
        for x in self.__dict__.values():
            if isinstance(x, qtype):
                while not x.empty():
                    _ = x.get()
        print('viewer stopped')

class DynamicArray2(object):
    def __init__(self, shape=3):
        if isinstance(shape, int):
            shape = (shape,)
        assert isinstance(shape, tuple)

        self.data = np.zeros((1000, *shape))
        self.shape = shape
        self.ind = 0

    def clear(self):
        self.ind = 0

    def append(self, x):
        self.extend([x])

    def append2(self, x):
        self.extend(x)

    def extend(self, xs):
        if len(xs) == 0:
            print('extend -> return ')
            return
        assert np.array(xs[0]).shape == self.shape

        if self.ind + len(xs) >= len(self.data):
            self.data.resize(
                (2 * len(self.data), *self.shape), refcheck=False)

        if isinstance(xs, np.ndarray):
            self.data[self.ind:self.ind + len(xs)] = xs
        else:
            for i, x in enumerate(xs):
                self.data[self.ind + i] = x
            self.ind += len(xs)

    def array(self):
        return self.data[:self.ind]

    def __len__(self):
        return self.ind

    def __getitem__(self, i):
        assert i < self.ind
        return self.data[i]

    def __iter__(self):
        for x in self.data[:self.ind]:
            yield x

class DynamicArray(object):
    def __init__(self, shape=3):
        if isinstance(shape, int):
            shape = (shape,)
        assert isinstance(shape, tuple)
        self.data = np.zeros((10000, *shape))
        self.shape = shape
        self.ind = 0

    def clear(self):
        self.ind = 0

    def append(self, x):
        self.extend([x])

    def extend(self, xs):
        if len(xs) == 0:
            return
        assert np.array(xs[0]).shape == self.shape

        if self.ind + len(xs) >= len(self.data):
            self.data.resize(
                (2 * len(self.data), *self.shape), refcheck=False)

        if isinstance(xs, np.ndarray):
            self.data[self.ind:self.ind + len(xs)] = xs
        else:
            for i, x in enumerate(xs):
                self.data[self.ind + i] = x
            self.ind += len(xs)

    def array(self):
        return self.data[:self.ind]

    def __len__(self):
        return self.ind

    def __getitem__(self, i):
        assert i < self.ind
        return self.data[i]

    def __iter__(self):
        for x in self.data[:self.ind]:
            yield x

