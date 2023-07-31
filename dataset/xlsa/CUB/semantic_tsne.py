from scipy import io
import numpy as np
from torch.nn import functional as F
from unseen_data_loader import data_loader_virtualCls
from unseen_option import Options
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


dataroot = './dataset/xlsa'
image_embedding = 'res101'
class_embedding = 'att'

matcontent = io.loadmat("C:/someproject/ZSLearning/AGZSL-main/dataset/xlsa/CUB/res101.mat")

feature = matcontent['features'].T
print(feature.shape)
label = matcontent['labels'].astype(int).squeeze() - 1
matcontent = io.loadmat("C:/someproject/ZSLearning/AGZSL-main/dataset/xlsa/CUB/att_splits.mat")

trainvalloc = matcontent['trainval_loc'].squeeze() - 1
test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

att_name = 'att'
attribute = matcontent[att_name].T

train_x = np.array(feature[trainvalloc])
train_label = np.array(label[trainvalloc])

test_x_seen = np.array(feature[test_seen_loc])
test_label_seen = np.array(label[test_seen_loc])

test_x_unseen = np.array(feature[test_unseen_loc])
test_label_unseen = np.array(label[test_unseen_loc])
print(train_x.shape)
print(train_label)
print(test_x_seen.shape)
print(test_label_seen)
print(test_x_unseen.shape)
print(test_label_unseen)
train_att = attribute[train_label]
print('Semantic: ', train_att.shape)
test_seen_att = attribute[test_label_seen]
print('Seen_att: ', test_seen_att.shape)
test_unseen_att = attribute[test_label_unseen]
print('Unseen_att: ', test_unseen_att.shape)
args = Options().parse()
ways = args.ways
shots = args.shots
# dataset = data_loader_virtualCls(train_x, train_att, train_label, ways=ways, shots=shots)
batch_att_concat = None
for i in range(30):
    dataset = data_loader_virtualCls(train_x, train_att, train_label, ways=ways, shots=shots)
    batch_visual, batch_att, batch_label = dataset.__getitem__(1)
    batch_visual = batch_visual.cuda()
    batch_visual_norm = F.normalize(batch_visual, p=2, dim=batch_visual.dim()-1, eps=1e-12)
    indx = torch.tensor(list(range(0, ways * shots, shots)))
    unique_batch_att = torch.index_select(batch_att, 0, indx).float().cuda()
    if i == 0:
        batch_att_concat = batch_att
    else:
        batch_att_concat = torch.cat((batch_att_concat, batch_att), dim=0)

# print(batch_visual.shape)
batch_att_concat = batch_att_concat.cpu().numpy()
print(batch_att_concat.shape)

# 合并数据集
features = np.concatenate((batch_att_concat, test_seen_att, test_unseen_att), axis=0)
# 对特征进行 t-SNE 降维，降到2维
tsne = TSNE(n_components=2, random_state=0)
# batch_att_concat_tsne = tsne.fit_transform(batch_att_concat)
# test_seen_tsne = tsne.fit_transform(test_seen_att)
# test_unseen_tsne = tsne.fit_transform(test_unseen_att)
features_tsne = tsne.fit_transform(features)

test_seen_start_idx = batch_att_concat.shape[0]
test_unseen_start_idx = test_seen_start_idx + batch_att_concat.shape[0]

# 降维后的坐标
test_seen_tsne = features_tsne[test_seen_start_idx:test_unseen_start_idx]
test_unseen_tsne = features_tsne[test_unseen_start_idx:]

# 绘制图像
plt.scatter(features_tsne[:test_seen_start_idx, 0], features_tsne[:test_seen_start_idx, 1], label='Virtual Classes', alpha=1)
plt.scatter(test_seen_tsne[:, 0], test_seen_tsne[:, 1], label='Seen Classes', alpha=0.7)
plt.scatter(test_unseen_tsne[:, 0], test_unseen_tsne[:, 1], label='Unseen Classes', alpha=0.4)
plt.legend()
plt.show()