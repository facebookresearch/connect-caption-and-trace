import numpy as np
import os
import h5py
import numpy as np
import jsonlines
import re
import json

# The first directory should lead to your feature files extracted by detectrons, and the box_only and feats_only are the new folders for saving bounding boxes and features (which will be used during training).

i = 0
for f in os.listdir('/mnt/m2/Datasets/ADE20k/full_images_feats/features/'):
    i += 1
    item = np.load('/mnt/m2/Datasets/ADE20k/full_images_feats/features/'+f)
    id = f.split('.jpg')[0]
    np.save('/mnt/m2/Datasets/ADE20k/full_images_feats/box_only/'+str(id), item['norm_bb'])
    np.savez('/mnt/m2/Datasets/ADE20k/full_images_feats/feats_only/'+str(id), item['box_feats'])
    if i % 1000 == 0:
        print('Processing #', i)
print('finish!')
