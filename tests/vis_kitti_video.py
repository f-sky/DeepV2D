import os.path as osp
import os

import cv2
from tqdm import tqdm
# from imageio import get_writer


def main():
    split_file = 'data/kitti/train_scenes_eigen.txt'
    with open(split_file)as f:
        lines = f.read().splitlines()
    for line in lines:
        d = '/home/linghao/Datasets/kitti/kitti_raw/%s/%s_sync/image_02/data' % (line[:10], line)
        files = sorted(os.listdir(d))
        for file in tqdm(files):
            img = cv2.imread(osp.join(d,file))
            cv2.imshow('frame', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
