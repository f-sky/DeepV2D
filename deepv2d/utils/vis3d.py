import numpy as np
import os
import shutil

import torch
import vis3d
from vis3d.vis3d.consts import get_world_transform
from vis3d.vis3d.exporter.exporter import FileHandler

from deepv2d.bounding_box_3d import Box3DList


class Vis3D(vis3d.Vis3D):
    has_removed = False  # ensure out_folder will be deleted once only when program starts.

    def __init__(self, xyz_pattern=('x', 'y', 'z'), out_folder='visual', sequence='sequence', flush_secs=0.001):
        self.rot_three_to_world = get_world_transform(xyz_pattern)
        self.rot_world_to_three = torch.inverse(self.rot_three_to_world)

        if not os.path.isabs(out_folder):
            out_folder = os.path.join(os.getcwd(), out_folder)
        if os.path.exists(out_folder) and not Vis3D.has_removed:
            shutil.rmtree(out_folder)
            Vis3D.has_removed = True
        out_folder = os.path.join(out_folder, sequence)

        self._file_handler = FileHandler(out_folder, flush_secs)

        self._scene_id = 0
        self.set_scene_id(0)

    def add_box3dlist(self, box3dlist: Box3DList, name=None):
        corners = box3dlist.convert('corners').bbox_3d.reshape(-1, 8, 3)
        self.add_boxes_by_corners(corners, name=name)

    def add_box3d(self, x, y, z, h, w, l, ry, name=None):
        b3d = Box3DList([[x, y, z, h, w, l, ry]])
        self.add_box3dlist(b3d, name)

    def add_label(self, label, name=None):
        self.add_box3d(label.x, label.y, label.z, label.h, label.w, label.l, label.ry, name)
