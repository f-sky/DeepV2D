import numpy as np
import os.path as osp
import os
import pickle

from tqdm import tqdm
from vis3d import Vis3D
from PIL import Image



def main():
    pred_dir = '/home/linghao/PycharmProjects/DeepV2D/kitti_image_depth_preds'
    with Vis3D(('x', '-y', '-z'), 'dbg') as vis3d:
        for file in tqdm(os.listdir(pred_dir)):
            vis3d.set_scene_id(int(file.split('.')[0]))
            img, depth, K = pickle.load(open(osp.join(pred_dir, file), 'rb'))
            # import pdb
            # pdb.set_trace()
            vis3d.add_image(Image.fromarray(img.astype(np.uint8)))
            pts = depth_to_rect(depth, K)
            vis3d.add_point_cloud(pts)


if __name__ == '__main__':
    main()
