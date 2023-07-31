import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from scipy import io
from torch.nn import functional as F
from unseen_data_loader import data_loader_virtualCls
from unseen_option import Options
from sklearn.preprocessing import StandardScaler

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
train_att = np.array(attribute[train_label])
print('Semantic: ', train_att.shape)
test_seen_att = np.array(attribute[test_label_seen])
print('Seen_att: ', test_seen_att.shape)
test_unseen_att = np.array(attribute[test_label_unseen])
print('Unseen_att: ', test_unseen_att.shape)
# 获得虚拟类
args = Options().parse()
ways = args.ways
shots = args.shots
dataset = data_loader_virtualCls(train_x, train_att, train_label, ways=ways, shots=shots)
batch_att_concat = None
for i in range(50):
    batch_visual, batch_att, batch_label = dataset.__getitem__(i)
    batch_visual = batch_visual.cuda()
    batch_visual_norm = F.normalize(batch_visual, p=2, dim=batch_visual.dim()-1, eps=1e-12)
    indx = torch.tensor(list(range(0, ways * shots, shots)))
    unique_batch_att = torch.index_select(batch_att, 0, indx).float().cuda()
    if i == 0:
        batch_att_concat = batch_att
    else:
        batch_att_concat = torch.cat((batch_att_concat, batch_att), dim=0)

batch_att_concat = batch_att_concat.cpu().numpy()
print(batch_att_concat.shape)

# 可视化
plt.rcParams['savefig.dpi'] = 600
test_unseen_num, _ = test_unseen_att.shape
test_seen_num, _ = test_seen_att.shape
sampled_num, _ = batch_att_concat.shape
train_test_data = np.concatenate([test_unseen_att, test_seen_att, batch_att_concat], axis=0)# [test_num+train_num+1+sampled_num, v]
print('start TSNE')
train_test_tsne = TSNE(n_components=2,init='random').fit_transform(train_test_data.data)
real_unseen_tsne = train_test_tsne[:test_unseen_num, :]
real_seen_tsne = train_test_tsne[test_unseen_num:test_unseen_num+test_seen_num, :]
sampled_tsne = train_test_tsne[test_unseen_num+test_seen_num:, :]
#plt.scatten(test_tsnel:0l.test tsnel:. 1..c=testclass labels..cmap=cmap,.s=3.manken='*'.alpha=0.3)
# #plt.scatten(train tsnel:,0l, train tsnel:, 1l.c=train class labels, cmap=cmap,.s=3,manken='.)
plt.scatter(sampled_tsne[:, 0],sampled_tsne[:, 1], color='#619505', s=1, linewidths=0, alpha=0.6, label='sampled_tsne') # tran sampled data, AWA2: s=3, linewidths=1, aloha-0.6"
plt.scatter(real_unseen_tsne[:, 0], real_unseen_tsne[:, 1], color='#f96441', s=1, linewidths=0, alpha=0.6, label='real_unseen_tsne') # test data,AWA2: s=4, linewidths=0, alpha=0.6
plt.scatter(real_seen_tsne[:, 0], real_seen_tsne[:, 1], color='#BE5A83', s=1, linewidths=0, alpha=0.6, label='real_seen_tsne') # test data,AWA2: s=4, linewidths=0, alpha=0.6
#plt.savefig('../CUB/unseen_feature_description.png', bbox_inches='tight')
plt.legend(markerscale=10)
plt.title('Semantic')
plt.show()
plt.close()