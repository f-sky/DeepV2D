from copy import copy

import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset


class SevenScenes(Dataset):

    def __len__(self) -> int:
        return len(self.test_data)

    def __init__(self, root_dir, scene_idx: int) -> None:
        super().__init__()
        self.db = LoadSevenScenes(root_dir)
        self.data_path = root_dir
        self.scene_idx = scene_idx
        self.scene, self.seq = self.db.test_seqs_list[scene_idx]
        self.scene_name = self.scene + '-' + self.seq
        self.file_paths = self.db.get_filepaths(self.scene, self.seq)
        num_imgs = len(self.file_paths)
        test_data = []
        for imageid_1 in range(10, num_imgs, 10):
            imageid_2 = imageid_1 + 10
            if imageid_2 < num_imgs:
                test_data.append((self.scene_name, imageid_1, imageid_2))
        self.test_data = test_data
        K = self.db.intrinsics
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)
        self.intrinsics = intrinsics

    def __getitem__(self, index):
        scanid, imageid_1, imageid_2 = self.test_data[index]

        num_frames = len(self.file_paths)

        images = []

        # we need to include imageid_2 and imageid_1 to compare to BA-Net poses,
        # then sample remaining 6 frames uniformly
        dt = imageid_2 - imageid_1
        s = 3
        poses = []
        for i in [0, dt, -3 * s, -2 * s, -s, s, 2 * s, 3 * s]:
            otherid = min(max(1, i + imageid_1), num_frames - 1)
            image, pose = self.db.load_sample(self.file_paths[otherid], 480, 640)
            # image_file = os.path.join(self.data_path, self.scene, 'images', "{:0>5d}.jpg".format(otherid))
            # image = cv2.imread(image_file)
            # image = cv2.resize(image, (640, 480))
            images.append(image)
            # pose = np.loadtxt(os.path.join(self.data_path, self.scene, 'poses', "{:0>5d}.txt".format(otherid)),
            #                   dtype='f', delimiter=' ')
            pose = np.linalg.inv(pose)  # todo:check?
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


class LoadSevenScenes(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.test_seqs_list = [('chess', 'seq-03'),
                               ('chess', 'seq-05'),
                               ('fire', 'seq-03'),
                               ('fire', 'seq-04'),
                               ('heads', 'seq-01'),
                               ('office', 'seq-02'),
                               ('office', 'seq-06'),
                               ('office', 'seq-07'),
                               ('office', 'seq-09'),
                               ('pumpkin', 'seq-01'),
                               ('pumpkin', 'seq-07'),
                               ('redkitchen', 'seq-03'),
                               ('redkitchen', 'seq-04'),
                               ('redkitchen', 'seq-06'),
                               ('redkitchen', 'seq-12'),
                               ('redkitchen', 'seq-14'),
                               ('stairs', 'seq-01'),
                               ('stairs', 'seq-04')]

        self.intrinsics = np.asarray([[585, 0, 320],
                                      [0, 585, 240],
                                      [0, 0, 1]])

    def get_filepaths(self, scene, seq):
        """ load list of filenames from one seq of scene;
        return filepaths of rgbs, depths, poses
        """
        seq_dir = os.path.join(self.root_dir, scene, seq)
        filepaths_list = []
        for filename in sorted(os.listdir(seq_dir)):
            if "color" in filename:
                rgb_name = filename
                # depth_name = rgb_name.replace("color", "depth")
                pose_name = rgb_name.replace("color.png", "pose.txt")
                # pred_depth_name = rgb_name.replace("color", "pred_depth")
                sample_path = {'rgb': os.path.join(seq_dir, rgb_name),
                               # 'depth': os.path.join(seq_dir, depth_name),
                               'pose': os.path.join(seq_dir, pose_name),
                               # 'pred_depth_name': pred_depth_name
                               }
                filepaths_list.append(sample_path)
        return filepaths_list

    def load_sample(self, sample_path, image_height_expected, image_width_expected):
        rgb = cv2.imread(sample_path['rgb'], -1)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # depth = cv2.imread(sample_path['depth'], -1) / 1000.0
        pose = self.read_pose_file(sample_path['pose'])

        extrinsics = self.pose2extrinsics(pose)
        cam = self.get_cam(self.intrinsics, extrinsics)

        original_image_width = rgb.shape[1]
        original_image_height = rgb.shape[0]
        scale_x = float(image_width_expected) / original_image_width
        scale_y = float(image_height_expected) / original_image_height

        cam = self.scale_cam(cam, scale_x, scale_y)
        rgb = self.scale_img(rgb, image_height_expected, image_width_expected, 'linear')
        # rgb = self.normalize_image(rgb)
        # depth = self.scale_img(depth, image_height_expected, image_width_expected, 'nearest')

        # rgb, depth, cam = self.toTensor(rgb, depth, cam)

        return rgb, cam

    # def toTensor(self, rgb, depth, cam):
    #     rgb = torch.from_numpy(np.float32(rgb))
    #     rgb = rgb.permute(2, 0, 1)  # (h,w,c) to (c, h, w)
    #
    #     depth = torch.from_numpy(np.float32(depth))
    #
    #     cam = torch.from_numpy(np.float32(cam))
    #
    #     return rgb, depth, cam

    def scale_img(self, image, expected_height, expected_width, interpolation):
        # although opencv load image in shape (height, width, channel), cv2.resize still need shape (width, height)
        if interpolation == 'linear':
            return cv2.resize(image, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
        if interpolation == 'nearest':
            return cv2.resize(image, (expected_width, expected_height), interpolation=cv2.INTER_NEAREST)
        if interpolation is None:
            raise Exception('interpolation cannot be None')

    def normalize_image(self, img):
        """
        Zero mean and Unit variance normalization to input image
        :param img: input image
        :return: normalized image
        """
        img = img / 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_normal = (img - mean) / std
        return img_normal.astype(np.float32)

    def read_pose_file(self, filepath):
        """Read in the pose/*.txt and parse into a ndarray"""
        pose = np.loadtxt(filepath, dtype='f', delimiter='\t ')

        return pose

    def pose2extrinsics(self, pose):
        """Convert pose(camera to world) to extrinsic matrix(world to camera)"""
        extrinsics = np.linalg.inv(pose)
        return extrinsics

    def get_cam(self, intrinsics, extrinsics):
        """convert to cam"""
        cam = np.zeros((2, 4, 4))

        # read extrinsic(world to camera) or pose(camera to pose)
        cam[0] = extrinsics

        # read intrinsic
        cam[1][0:3, 0:3] = self.intrinsics

        return cam

    def scale_cam(self, cam, scale_x, scale_y):
        """scale the camera intrinsics of one view"""
        new_cam = copy(cam)
        # focal
        new_cam[1][0][0] *= scale_x
        new_cam[1][1][1] *= scale_y

        # principle point:
        new_cam[1][0][2] *= scale_x
        new_cam[1][1][2] *= scale_y

        return new_cam


def main():
    ss = SevenScenes('/data/7scenes', 0)
    print('len', len(ss))
    print(ss[0])


if __name__ == '__main__':
    main()