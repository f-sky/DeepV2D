import os.path as osp
import os

import cv2
import numpy as np


class DemoDataset:
    """Pytorch Dataset for a single scene. getitem loads individual frames"""

    def __init__(self, data_path, scene):
        """
        Args:
        """
        self.scene = scene
        self.data_path = data_path
        # self.max_depth = max_depth
        self.size = (640, 480)
        num_imgs = len(os.listdir(osp.join(self.data_path, self.scene, 'images')))
        test_data = []
        for imageid_1 in range(10, num_imgs, 10):
            imageid_2 = imageid_1 + 10
            if imageid_2 < num_imgs:
                test_data.append((scene, imageid_1, imageid_2))
        self.test_data = test_data

        K = np.loadtxt(os.path.join(self.data_path, self.scene, 'intrinsics', "00000.txt"),
                       dtype='f', delimiter=' ')
        # w, h = 1920, 1440
        # K[0, :] /= (w / self.size[0])
        # K[1, :] /= (h / self.size[1])
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)
        self.intrinsics = intrinsics

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        scanid, imageid_1, imageid_2 = self.test_data[index]
        scandir = os.path.join(self.data_path, self.scene)
        num_frames = len(os.listdir(os.path.join(scandir, 'images')))

        images = []

        # we need to include imageid_2 and imageid_1 to compare to BA-Net poses,
        # then sample remaining 6 frames uniformly
        dt = imageid_2 - imageid_1
        s = 3
        poses = []
        for i in [0, dt, -3 * s, -2 * s, -s, s, 2 * s, 3 * s]:
            otherid = min(max(1, i + imageid_1), num_frames - 1)
            # image_file = os.path.join(scandir, 'color', '%d.jpg' % otherid)
            image_file = os.path.join(self.data_path, self.scene, 'images', "{:0>5d}.jpg".format(otherid))
            image = cv2.imread(image_file)
            image = cv2.resize(image, (640, 480))
            images.append(image)
            pose = np.loadtxt(os.path.join(self.data_path, self.scene, 'poses', "{:0>5d}.txt".format(otherid)),
                              dtype='f', delimiter=' ')
            pose = np.linalg.inv(pose)
            poses.append(pose)
        poses = np.stack(poses)

        # depth_file = os.path.join(scandir, 'depth', '%d.png' % imageid_1)
        # depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        # depth = (depth / 1000.0).astype(np.float32)

        pose1 = np.loadtxt(os.path.join(self.data_path, self.scene, 'poses', "{:0>5d}.txt".format(imageid_1)),
                           dtype='f', delimiter=' ')
        pose2 = np.loadtxt(os.path.join(self.data_path, self.scene, 'poses', "{:0>5d}.txt".format(imageid_2)),
                           dtype='f', delimiter=' ')
        pose1 = np.linalg.inv(pose1)
        pose2 = np.linalg.inv(pose2)
        pose_gt = np.dot(pose2, np.linalg.inv(pose1))

        images = np.stack(images, axis=0).astype(np.uint8)
        # depth = depth.astype(np.float32)

        data_blob = {
            'images': images,
            # 'depth': depth,
            'pose': pose_gt,
            'poses': poses,
            'intrinsics': self.intrinsics,
            "scanid": scanid,
            "imageid_1": imageid_1,
            "imageid_2": imageid_2
        }

        return data_blob
