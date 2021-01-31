import numpy as np


def load_test_scanids():
    test_frames = np.loadtxt('data/scannet/scannet_test.txt', dtype=np.unicode_)
    scans = set()
    for i in range(0, len(test_frames), 4):
        test_frame_1 = str(test_frames[i]).split('/')
        scan = test_frame_1[3]
        scans.add(scan)
    return scans
