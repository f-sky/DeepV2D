import matplotlib.pyplot as plt
import pickle

batch = pickle.load(open('tmp/batch.pkl', 'rb'))
id_batch, images_batch, poses_batch, gt_batch, filled_batch, pred_batch, intrinsics_batch = batch
print()
idx = 3
# for img in images_batch[idx]:
#     plt.imshow(img / 255.0)
#     plt.show()
plt.imshow(images_batch[idx, 0][:,:,::-1]/255.0)
plt.show()
plt.imshow(gt_batch[idx, :, :, 0], 'jet')
plt.show()
plt.imshow(filled_batch[idx, :, :, 0], 'jet')
plt.show()
print(poses_batch[idx])
