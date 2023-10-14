"""
Part of code is from Kai Li "kailigo". The gitub website is https://github.com/kailigo/cvcZSL.
We add virtual classes and IAS in the code.
"""
from scipy import io
import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import lr_scheduler
import torch.utils.data as data
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter

from utils import ReDirectSTD
from unseen_data_loader import data_loader_virtualCls
from unseen_option import Options

import os
import random
import pickle
# from test_embeded import test_while_training_simple
import datetime
import argparse

TMP = 10
# 指定运行GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
args = Options().parse()
model_file_name = './chk/' + args.model_file
summaryFolder = './summary/' + args.log_file
if not os.path.exists('./chk'):
    os.mkdir('./chk')
# 获取当前日期
now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d")
summaryFile = './summary/record/'
# if not os.path.exists(summaryFolder):
#     # os.mkdir(summaryFolder)
#     # 修改，上面代码会报错找不到指定路径
#     os.makedirs(summaryFolder)
if not os.path.exists(summaryFile):
    # os.mkdir(summaryFolder)
    # 修改，上面代码会报错找不到指定路径
    # os.makedirs(summaryFile)
    os.makedirs(summaryFile)
# 设置文件名和后缀
name = args.dataset
filename = f"{name}_{date}.txt"
suffix = 0
# 如果文件已经存在，则增加一个数字后缀
while os.path.isfile(summaryFile + filename):
    suffix += 1
    filename = f"{name}_{date}_{suffix}.txt"

writer = SummaryWriter(summaryFolder)

print(args)

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    # torch cuda
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

init_seeds(520)
def calc_accuracy(test_visual, test_label, attM, test_id, test_id_seen_unseen,cossim=False):       
        outpred = [0] * test_visual.shape[0]
        end = 0
        outpred_list = []
        zslpred_list = []
        score_n_list = []
        for j in range(0,len(test_visual),64):
            if j+64>len(test_visual):
                end = len(test_visual)
            else:
                end = j+64
            all_cls_weights = forward(attM,test_visual[j:end],tmp=TMP)       
            score,score_n=apply_classification_weights(test_visual[j:end].cuda(), 
                    all_cls_weights,norm=True)
            score = score.squeeze(0)
            score_n = score_n.squeeze(0)
            
            _, pred = score.max(dim=1)
            pred = pred.view(-1)
            select_test_label = test_label[j:end].view(-1)

            outpred_list.extend(test_id[pred.cpu().detach().numpy()])
            zslpred_list.extend([ i in test_id_seen_unseen for i in test_id[pred.cpu().detach().numpy()]])
            score_n_list.extend(score_n.cpu().detach().numpy())

        seen_unseen_acc = accuracy_score(np.ones(len(zslpred_list)),zslpred_list)
        
        outpred = np.array(outpred_list, dtype='int')
        score_n = np.array(score_n_list)
        test_label = test_label.numpy()
        unique_labels = np.unique(test_label)
        acc = 0
        acc_cls = {}
        preds_cls = {}
        cos_cls = {}
        for l in unique_labels:
                idx = np.nonzero(test_label == l)[0]
                acc_cls[l] = accuracy_score(test_label[idx],outpred[idx])
                try:
                    preds_cls[l].extend(list(outpred[idx])) 
                except(KeyError):
                    preds_cls[l] = list(outpred[idx])

                acc += acc_cls[l]
                loc = test_label == l
                outpred_l = outpred[idx]
                score_n_l = score_n[idx]
                loc = outpred_l == l

                if cossim:
                    cos_c = score_n_l[loc,l]
                    try:
                        cos_cls[l].extend(cos_c)
                    except(KeyError):
                        cos_cls[l] = [cos_c]
        acc = acc / unique_labels.shape[0]

        return acc,seen_unseen_acc,acc_cls,cos_cls,preds_cls


def compute_accuracy_all(test_att, att_all, test_visual_unseen, test_id_unseen, test_label_unseen,
                test_visual_seen, test_id_all, test_label_seen, train_id):

        acc_zsl,_,unseenacc_cls,_,unseenpred_cls = calc_accuracy(test_visual_unseen, test_label_unseen, test_att, test_id_unseen,test_id_unseen)
        acc_seenAcc,_,seenacc_cls,_,_ = calc_accuracy(test_visual_seen, test_label_seen, att_all, train_id, train_id)

        att_all_cls = torch.cat((att_all, test_att))
        
        acc_gzsl_unseen,Ru,unseengeneralAcc_cls,unseenCos_cls,_ = calc_accuracy(test_visual_unseen, test_label_unseen, att_all_cls, test_id_all,test_id_unseen,cossim=True)
        
        acc_gzsl_seen,Rs,seengeneralAcc_cls,seenCos_cls,_ = calc_accuracy(test_visual_seen, test_label_seen, att_all_cls, test_id_all,train_id,cossim=True)       
        acc_cls = {**unseenacc_cls,**seenacc_cls}
        generalAcc_cls = {**unseengeneralAcc_cls,**seengeneralAcc_cls}
        H = 2 * acc_gzsl_seen * acc_gzsl_unseen / (acc_gzsl_seen + acc_gzsl_unseen)

        return acc_zsl, acc_seenAcc, acc_gzsl_unseen, acc_gzsl_seen, H, Rs, Ru, acc_cls, generalAcc_cls,unseenCos_cls,seenCos_cls,unseenpred_cls


def apply_classification_weights(features, cls_weights,norm=False):

        features = F.normalize(features,dim=-1)
        cls_weights = F.normalize(cls_weights, p=2, dim=-1, eps=1e-12) 

        cls_scores = scale_cls * (torch.matmul(cls_weights,features.t()))
        cls_scores = cls_scores.permute(0,2,1)
        cls_scores = torch.diagonal(cls_scores,offset=0,dim1=0,dim2=1)
        cls_scores = cls_scores.t()
        if norm:
            return cls_scores,cls_scores/scale_cls
        else:
            return cls_scores


def IASatt(features,AttM,tmp):
        cls_num = len(AttM)
        attdims = AttM.shape[1]
        atten = torch.mm(features,w_IAS)+b_IAS
        atten = F.softmax(atten/tmp,dim=1).reshape(-1,1,attdims)
        atten = atten.reshape(-1,1,attdims)
        atten = atten + torch.ones_like(atten)
        atten = atten.repeat(1,cls_num,1)
        AttM = AttM.unsqueeze(0)
        AttM = AttM.repeat(len(features),1,1)
        AttM = atten*AttM
        return AttM

def forward(att,features,tmp):
        features = features.squeeze()
        att = IASatt(features,att,tmp)

        a1 = F.relu(torch.matmul(att, w1) + b1)
        a2 = F.relu(torch.matmul(a1, w2) + b2)

        return a2

# scaler = preprocessing.MinMaxScaler()
dataroot = '/data/xbjin_data/lw/ZSLearning/AGZSL-main/dataset/xlsa/'
image_embedding = 'res101' 
class_embedding = 'att'
dataset = args.dataset
matcontent = io.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")

feature = matcontent['features'].T
label = matcontent['labels'].astype(int).squeeze() - 1
matcontent = io.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")

trainvalloc = matcontent['trainval_loc'].squeeze() - 1
test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

att_name = 'att'
attribute = matcontent[att_name].T

#标准化
# _train_feature = scaler.fit_transform(feature[trainvalloc])
# _test_seen_feature = scaler.transform(feature[test_seen_loc])
# _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
# train_feature = torch.from_numpy(_train_feature).float()
# mx = train_feature.max()
# train_feature.mul_(1/mx)
# train_label = torch.from_numpy(label[trainvalloc]).long()
# test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
# test_unseen_feature.mul_(1/mx)
# test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
# test_seen_feature = torch.from_numpy(_test_seen_feature).float()
# test_seen_feature.mul_(1/mx)
# test_seen_label = torch.from_numpy(label[test_seen_loc]).long()


clsname = [ matcontent['allclasses_names'][i][0][0] for i in range(len(matcontent['allclasses_names']))]
train_x = np.array(feature[trainvalloc])
train_label = np.array(label[trainvalloc])

test_x_seen = np.array(feature[test_seen_loc])
test_label_seen = np.array(label[test_seen_loc])

test_x_unseen = np.array(feature[test_unseen_loc])
test_label_unseen = np.array(label[test_unseen_loc])


train_att = attribute[train_label]
train_id, idx = np.unique(train_label, return_inverse=True)
train_att_unique = attribute[train_id]


test_id, idx = np.unique(test_label_unseen, return_inverse=True)
att_pro = attribute[test_id]
train_test_id = np.concatenate((train_id, test_id))

_, idx = np.unique(test_label_seen, return_inverse=True)

att_dim = train_att.shape[1]
feat_dim = train_x.shape[1]

att_pro = torch.from_numpy(att_pro).float().cuda()
#train_x是后加的
train_x = torch.from_numpy(train_x).float().cuda()
train_x = F.normalize(train_x, p=2, dim=train_x.dim()-1, eps=1e-12)
test_x_seen = torch.from_numpy(test_x_seen).float().cuda()
test_x_seen = F.normalize(test_x_seen, p=2, dim=test_x_seen.dim()-1, eps=1e-12)
test_x_unseen = torch.from_numpy(test_x_unseen).float().cuda()
test_x_unseen = F.normalize(test_x_unseen, p=2, dim=test_x_unseen.dim()-1, eps=1e-12)
test_label_seen = torch.tensor(test_label_seen)
test_label_unseen = torch.tensor(test_label_unseen)

att_all = torch.from_numpy(train_att_unique).float().cuda()

bias = nn.Parameter(torch.FloatTensor(1).fill_(0).cuda(), requires_grad=True)
scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10).cuda(), requires_grad=True)
w1 = Variable(torch.FloatTensor(att_dim, args.hidden_dim).cuda(), requires_grad=True)
b1 = Variable(torch.FloatTensor(args.hidden_dim).cuda(), requires_grad=True)
w2 = Variable(torch.FloatTensor(args.hidden_dim, 2048).cuda(), requires_grad=True)
b2 = Variable(torch.FloatTensor(2048).cuda(), requires_grad=True)
w_IAS = Variable(torch.FloatTensor(2048, att_dim).cuda(), requires_grad=True)
b_IAS = Variable(torch.FloatTensor(att_dim).cuda(), requires_grad=True)

w1.data.normal_(0, 0.02)
w2.data.normal_(0, 0.02)
b1.data.fill_(0)
b2.data.fill_(0)
w_IAS.data.normal_(0,0.02)
b_IAS.data.fill_(0)

# New_network
from new_network import new_network
from new_network import attNetwork
netN = new_network(args)
netN.cuda()

netA = attNetwork(args)
netA.cuda()

#develop network
# from new_network import RelationNet
# from new_network import Discriminator
# from new_network import AE
# relationNet = RelationNet(args)
# relationNet.cuda()
# discriminator = Discriminator(args)
# discriminator.cuda()
# ae = AE(args)
# ae.cuda()

# add our net parameter
optimizer = torch.optim.Adam([w_IAS,b_IAS,w1, b1, w2, b2, bias, scale_cls], lr=args.lr, weight_decay=args.opt_decay)

# New_network
optimizerN = torch.optim.Adam(netN.parameters(), lr=args.lr)
optimizerA = torch.optim.Adam(netA.parameters(), lr=args.lr)

# develop network
# relationNet_optimizer = torch.optim.Adam(relationNet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# ae_optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# ones = torch.ones(args.batchSize, dtype=torch.long).cuda()
# zeros = torch.zeros(args.batchSize, dtype=torch.long).cuda()

# breakpoint()
step_size = args.step_size

gamma = args.gamma
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
criterion = nn.CrossEntropyLoss()

ways = args.ways
shots = args.shots

seen_classes = torch.from_numpy(np.unique(train_label))
dataset = data_loader_virtualCls(train_x, train_att, train_label, seen_classes, ways=ways, shots=shots)

# breakpoint()
best_acc_zsl = 0.0
best_acc_gzsl_seen = 0.0
best_acc_gzsl_unseen = 0.0
best_H = 0.0
best_epoch = 0
best_unseenAcc = 0.0



# 打开txt记录训练信息
fb = open(summaryFile + filename, 'w')
description=str(args)
fb.write(description + '\n')
# 训练新网络的loss
for pre_epoch in range(args.pre_epochs):
    pre_epoch_loss = 0
    loss_visual_to = 0
    loss_att_to = 0
    loss_visual_mse_vatt_total = 0
    print("%d epoch" % (pre_epoch))
    for i in range(500):
        pre_loss, loss_visual, loss_att, loss_visual_mse_vatt = dataset.__train_newnet__(pre_epoch, netN, netA, optimizerN, optimizerA)
        print("%d/500 steps,loss = %.4f, loss_visual = %.4f, loss_att = %.4f, loss_visual_mse_vatt = %.4f" % (i, pre_loss.item(), loss_visual.item(), loss_att.item(), loss_visual_mse_vatt.item()))
        # pre_loss = torch.tensor(pre_loss)
        pre_epoch_loss = pre_epoch_loss + pre_loss
        loss_visual_to = loss_visual_to + loss_visual
        loss_att_to = loss_att_to + loss_att
        loss_visual_mse_vatt_total = loss_visual_mse_vatt_total + loss_visual_mse_vatt
    pre_epoch_loss = pre_epoch_loss / 500
    loss_visual_to = loss_visual_to / 500
    loss_att_to = loss_att_to / 500
    loss_visual_mse_vatt_total = loss_visual_mse_vatt_total / 500
    print("loss: %.4f, loss_visual: %.4f, loss_att: %.4f, loss_visual_mse_vatt = %.4f" % (pre_epoch_loss, loss_visual_to, loss_att_to, loss_visual_mse_vatt_total))
    con = ('ep: %d, loss: %.4f, loss_visual = %.4f, loss_att = %.4f, loss_visual_mse_vatt = %.4f' % (pre_epoch, pre_epoch_loss, loss_visual_to, loss_att_to, loss_visual_mse_vatt_total))
    fb.write(con + '\n')
#     # 保存训练模型
    if(pre_epoch + 1) % 10 == 0:
        model_save_path = f"pre_models/CUB_modify_para/model_pre_epoch_{pre_epoch + 1}.pt"
        torch.save(netN.state_dict(), model_save_path)
        print(f"Saved model for epoch {pre_epoch} at {model_save_path}")

# # 加载已经保存的loss模型
# model_path = "/data/xbjin_data/lw/ZSLearning/AGZSL-main/pre_models/CUB_modify_para/model_pre_epoch_160.pt"
# netN.load_state_dict(torch.load(model_path))

# SDGZSL's config
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default='SUN',help='dataset: CUB, AWA2, APY, FLO, SUN')
# parser.add_argument('--dataroot', default='./SDGZSL_data', help='path to dataset')
# parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
# parser.add_argument('--image_embedding', default='res101', type=str)
# parser.add_argument('--class_embedding', default='att', type=str)
#
# parser.add_argument('--gen_nepoch', type=int, default=400, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train generater')
#
# parser.add_argument('--zsl', type=bool, default=False, help='Evaluate ZSL or GZSL')
# parser.add_argument('--finetune', type=bool, default=False, help='Use fine-tuned feature')
# parser.add_argument('--ga', type=float, default=15, help='relationNet weight')
# parser.add_argument('--beta', type=float, default=1, help='tc weight')
# parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
# parser.add_argument('--dis', type=float, default=3, help='Discriminator weight')
# parser.add_argument('--dis_step', type=float, default=2, help='Discriminator update interval')
# parser.add_argument('--kl_warmup', type=float, default=0.01, help='kl warm-up for VAE')
# parser.add_argument('--tc_warmup', type=float, default=0.001, help='tc warm-up')
#
# parser.add_argument('--vae_dec_drop', type=float, default=0.5, help='dropout rate in the VAE decoder')
# parser.add_argument('--vae_enc_drop', type=float, default=0.4, help='dropout rate in the VAE encoder')
# parser.add_argument('--ae_drop', type=float, default=0.2, help='dropout rate in the auto-encoder')
#
# parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
# parser.add_argument('--classifier_steps', type=int, default=50, help='training steps of the classifier')
#
# parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
# parser.add_argument('--nSample', type=int, default=1200, help='number features to generate per class')
#
# parser.add_argument('--disp_interval', type=int, default=200)
# parser.add_argument('--save_interval', type=int, default=10000)
# parser.add_argument('--evl_interval',  type=int, default=400)
# parser.add_argument('--evl_start',  type=int, default=0)
# parser.add_argument('--manualSeed', type=int, default=5606, help='manual seed')
#
# parser.add_argument('--latent_dim', type=int, default=20, help='dimention of latent z')
# parser.add_argument('--q_z_nn_output_dim', type=int, default=128, help='dimention of hidden layer in encoder')
# parser.add_argument('--S_dim', type=int, default=1024)
# parser.add_argument('--NS_dim', type=int, default=1024)
#
# parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
# opt = parser.parse_args()
# opt.Z_dim = opt.latent_dim
# opt.X_dim = train_x.shape[1]
# opt.C_dim = attribute.shape[1]
# from models import VAE
# # netM = VAE(opt)
# # netM.cuda()
# model_path = '/data/xbjin_data/lw/ZSLearning/SDGZSL-main/out/CUB/wd-1e-08_b-0.003_g-5_lr-0.0001_sd-2048_dis-0.3_nS-1000_nZ-20_bs-64_CUB_H_modify2Best_model_save_H_50.41_S_47.39_U_53.85.pth'
# netM = torch.load(model_path, map_location='cuda:0')
# netM.eval()
# from models import AE
# # netAE = AE(opt)
# # netAE.cuda()
# ae_path = '/data/xbjin_data/lw/ZSLearning/SDGZSL-main/out/CUB/wd-1e-08_b-0.003_g-5_lr-0.0001_sd-2048_dis-0.3_nS-1000_nZ-20_bs-64_CUB_H_modify2Best_ae_save_H_50.41_S_47.39_U_53.85.pth'
# netAE = torch.load(ae_path, map_location='cuda:0')
# netAE.eval()

netN.eval()
netA.eval()

for epoch in range(args.num_epochs):
        epoch_loss = 0
        lr_scheduler.step()

        for i in range(1000):
                batch_visual, batch_att, batch_label = dataset.__our_getitem__(i, netN) # __our_getitem__(i,ournet) use our net to process att and visual, and new att and visual
                # batch_visual, batch_att, batch_label = dataset.__getitem__(i)
                batch_visual = batch_visual.cuda()
                batch_visual_norm = F.normalize(batch_visual, p=2, dim=batch_visual.dim()-1, eps=1e-12)                         

                indx = torch.tensor(list(range(0, ways*shots, shots)))  
                unique_batch_att = torch.index_select(batch_att, 0, indx).float().cuda()                

                all_cls_weights = forward(unique_batch_att,batch_visual_norm,tmp=TMP)

                score = apply_classification_weights(batch_visual_norm, all_cls_weights)
                score = score.squeeze(0)
                # design our loss...
                loss = criterion(score, Variable(batch_label.cuda())) # + our_loss for our net according by Bias-Eliminated Semantic Refinement for Any-Shot Learning
                print("%d/1000 steps,loss = %.4f"%(i,loss.item()))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([w_IAS,b_IAS,w1, b1, w2, b2, scale_cls, bias], 1)
                optimizer.step()
                epoch_loss = epoch_loss + loss

        epoch_loss = epoch_loss / 1000.0
        epoch_loss = epoch_loss.data.cpu().numpy()

        acc_zsl, seenAcc, acc_unseen_gzsl, acc_seen_gzsl, H, Rs,Ru,accs_cls,generalAccs_cls,unseenCos_cls,seenCos_cls,unseenpred_cls = compute_accuracy_all(att_pro, att_all, test_x_unseen, 
                test_id, test_label_unseen, test_x_seen, train_test_id,  test_label_seen, train_id)
        
        H = 2 * acc_seen_gzsl * acc_unseen_gzsl / (acc_seen_gzsl + acc_unseen_gzsl)
        print(H)
        writer.add_scalar('general/acc_seen_gzsl',acc_seen_gzsl,epoch)
        writer.add_scalar('general/acc_unseen_gzsl',acc_unseen_gzsl,epoch)
        writer.add_scalar('general/H',H,epoch)
        writer.add_scalar('split/unseenAcc',acc_zsl,epoch)
        writer.add_scalar('split/seenAcc',seenAcc,epoch)
        writer.add_scalar('split/Rs',Rs,epoch)
        writer.add_scalar('split/Ru',Ru,epoch)
        writer.add_scalar('loss/loss',epoch_loss,epoch)
        # 写入训练信息
        content = ('ep: %d,  loss: %.4f,  zsl: %.4f, seenAcc: %.4f  gzsl: seen=%.4f, unseen=%.4f, h=%.4f, Rs=%.4f, Ru=%.4f ' %
                        (epoch,  epoch_loss, acc_zsl, seenAcc, acc_seen_gzsl, acc_unseen_gzsl, H, Rs, Ru,))
        fb.write(content + '\n')

        if acc_zsl > best_unseenAcc:
                print('save best acc')
                best_unseenAcc = acc_zsl
                best_epoch = epoch
                best_acc_zsl = acc_zsl          
                best_seen_acc = seenAcc          
                best_acc_gzsl_seen = acc_seen_gzsl
                best_acc_gzsl_unseen = acc_unseen_gzsl
                best_H = H
                best_Ru = Ru
                best_Rs = Rs

                best_w1 = w1.data.clone()
                best_b1 = b1.data.clone()
                best_w2 = w2.data.clone()
                best_b2 = b2.data.clone()
                best_scale_cls = scale_cls.data.clone()
                best_bias = bias.data.clone()

                torch.save({'w1': best_w1, 'b1': best_b1, 'w2': best_w2, 'b2': best_b2, 
                    'scale_cls': best_scale_cls, 'bias': best_bias,'w_IAS':w_IAS,'b_IAS':b_IAS}, model_file_name.replace('.pt','bestunseenAcc.pt'))
                

        for param_group in optimizer.param_groups:
                print('ep: %d,  lr: %lf, loss: %.4f,  zsl: %.4f, seenAcc: %.4f  gzsl: seen=%.4f, unseen=%.4f, h=%.4f, Rs=%.4f, Ru=%.4f ' % 
                        (epoch, param_group['lr'],  epoch_loss, acc_zsl, seenAcc, acc_seen_gzsl, acc_unseen_gzsl, H, Rs, Ru,))

print(model_file_name)
print('best_ep: %d, zsl: %.4f, seenAcc: %.4f  gzsl: seen=%.4f, unseen=%.4f, h=%.4f, Rs=%.4f, Ru=%.4f' % 
        (best_epoch, best_acc_zsl,best_seen_acc, best_acc_gzsl_seen, best_acc_gzsl_unseen, best_H, best_Rs, best_Ru))

fb.write(('best_ep: %d, zsl: %.4f, seenAcc: %.4f  gzsl: seen=%.4f, unseen=%.4f, h=%.4f, Rs=%.4f, Ru=%.4f' %
        (best_epoch, best_acc_zsl,best_seen_acc, best_acc_gzsl_seen, best_acc_gzsl_unseen, best_H, best_Rs, best_Ru)) + '\n')
fb.close()
