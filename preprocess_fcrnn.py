import os
import numpy as np
import sys
import pickle
import time

dir_np_chargrid_1h = "/content/data/outdir_np_chargrid_1h/"
dir_np_chargrid = "/content/data/outdir_np_chargrid/"
dir_np_gt_1h = "/content/data/outdir_np_gt_1h/"
dir_np_bbox_anchor_mask = "/content/data/outdir_np_bbox_anchor_mask/"
dir_np_bbox_anchor_coord = "/content/data/outdir_np_bbox_anchor_coord/"

if not os.path.exists(dir_np_chargrid):
    os.makedirs(dir_np_chargrid)

print("Number of chargrid files: " + str(len(os.listdir(dir_np_chargrid_1h))))
print("Number of segmentation mask files: " + str(len(os.listdir(dir_np_gt_1h))))
print("Number of anchor mask files: " + str(len(os.listdir(dir_np_bbox_anchor_mask))))
print("Number of bounding box anchor files: " + str(len(os.listdir(dir_np_bbox_anchor_coord))))

class_mapping = {0: 'Total',
                 1: 'Address',
                 2: 'Company name',
                 3: 'Date'}


def remove_key(d, key):
    r = dict(d)
    del r[key]
    return r


train_data = []

st = time.time()
for each in os.listdir(dir_np_chargrid_1h):
    img_1h = np.load(os.path.join(dir_np_chargrid_1h, each))
    img = np.argmax(img_1h, axis=2)
    out = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    out[:, :, 0] = img
    out[:, :, 1] = img
    out[:, :, 2] = img
    if not os.path.exists(os.path.join(dir_np_chargrid, each)):
        sys.stdout.write('\r' + 'Saving ' + each + ' as image and processing')
        np.save(os.path.join(dir_np_chargrid, each), out)
    else:
        sys.stdout.write('\r' + 'Processing ' + each)
    bboxes = []
    anchor_mask = np.load(os.path.join(dir_np_bbox_anchor_mask, each))
    anchor_coord = np.load(os.path.join(dir_np_bbox_anchor_coord, each))
    anchor_mask = anchor_mask.astype(int)
    for i in range(0, anchor_mask.shape[0]):
        for j in range(0, anchor_mask.shape[1]):
            for k in range(0, 4):
                if anchor_mask[i][j][2 * k] == 1:
                    class_id = k
                    bbox_x1 = int(anchor_coord[i][j][4 * k] * out.shape[1])
                    bbox_y1 = int(anchor_coord[i][j][4 * k + 1] * out.shape[0])
                    bbox_x2 = int(anchor_coord[i][j][4 * k + 2] * out.shape[1])
                    bbox_y2 = int(anchor_coord[i][j][4 * k + 3] * out.shape[0])
                    bbox_hash = class_id + bbox_x1 + bbox_x2
                    if not any(d['bbox_hash'] == bbox_hash for d in bboxes):
                        bboxes.append({'class': class_mapping[class_id],
                                       'x1': bbox_x1,
                                       'x2': bbox_x2,
                                       'y1': bbox_y1,
                                       'y2': bbox_y2,
                                       'bbox_hash': bbox_hash})
    bboxes = [remove_key(d, 'bbox_hash') for d in bboxes]
    train_data.append({'filepath': os.path.join(dir_np_chargrid, each),
                       'width': out.shape[1],
                       'height': out.shape[0],
                       'bboxes': bboxes,
                       'imageset': 'trainval'})
print()
print('Spent %0.2f mins to load the data' % ((time.time() - st) / 60))

with open('./data/train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)
