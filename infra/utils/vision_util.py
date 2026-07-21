import numpy as np
import open3d as o3d
import threading
import cv2


def depth_to_point_cloud(depth_m: np.ndarray, fx, fy, cx, cy):
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_m
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z

    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k], axis=1
            )

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


class PointCloudDisplayer:
    def __init__(self, points: np.ndarray, left=100, top=100, width=640, height=480):
        self.window = o3d.visualization.Visualizer()
        self.window.create_window(
            window_name="Point Cloud",
            height=height,
            width=width,
            visible=True,
            left=left,
            top=top,
        )
        opt = self.window.get_render_option()
        opt.background_color = np.array([1.0, 1.0, 1.0])
        opt.point_size = 3.0
        opt.show_coordinate_frame = True
        self.pc = o3d.geometry.PointCloud()
        self.pc.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        self.window.add_geometry(self.pc)
        self.window.add_geometry(self.coord_frame)
        self._first_frame = True

    def display(self, points: np.ndarray):
        self.pc.points = o3d.utility.Vector3dVector(
            points[:, :3].astype(np.float64)
        )
        if self._first_frame:
            self.window.reset_view_point(True)
            self._first_frame = False
        self.window.update_geometry(self.pc)
        self.window.poll_events()
        self.window.update_renderer()

    def close(self):
        self.window.destroy_window()