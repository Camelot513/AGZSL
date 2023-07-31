import numpy as np
from scipy import io

chkFile = '/data/xbjin_data/lw/ZSLearning/AGZSL-main/dataset/xlsa/SUN/res101.mat'

matfile = io.loadmat(chkFile)
all_image = matfile['image_files']
all_features = matfile['features'].T
all_labels = matfile['labels']
print(matfile.keys())
print(matfile)
print(all_image.shape)
print(all_features.shape)
print(all_labels.shape)

chkFile = '/data/xbjin_data/lw/ZSLearning/AGZSL-main/dataset/xlsa/SUN/att_splits.mat'

attmat = io.loadmat(chkFile)
print(attmat.keys())
print(attmat)
allclass = np.unique(attmat['allclasses_names'])
print(allclass.shape)
att = attmat['att'].T
print(att.shape)
test_seen_idx = attmat['test_seen_loc'].squeeze() - 1
print(all_features.shape)
test_seen_features = all_features[test_seen_idx]
print(test_seen_idx.shape)
print(test_seen_features.shape)
print(attmat['test_unseen_loc'].shape)
print(attmat['trainval_loc'].shape)

