"""
Part of this code is from Kai Li "kailigo". The website is https://github.com/kailigo/cvcZSL.
"""
import math
from scipy import io
import numpy as np
import torch
from pdb import set_trace as breakpoint
import torch.utils.data as data
from sklearn.metrics import mean_squared_error
from unseen_option import Options
from utils import permute_dims
import torch.nn.functional as F

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label

class data_loader_virtualCls(data.Dataset):
        def __init__(self, feats, atts, labels, seen_classes, ways=16, shots=4):
                self.ways = ways*2                
                self.shots = shots    

                self.feats = torch.tensor(feats).float()
                # self.feats = feats
                self.atts = torch.tensor(atts).float()
                self.labels = labels
                self.classes = np.unique(labels)

                self.seen_classes = seen_classes
        def __getitem__(self, index):
                is_first = True
                select_feats = []
                select_atts = []
                select_labels = []        
                select_labels = torch.LongTensor(self.ways*self.shots)
                selected_classes = np.random.choice(list(self.classes), self.ways, False) # AWA2和APY需要把这里改为True,CUN和SUN可以是False
                #selected_classes = self.classes
                #mixup
                cls_idx = {}


                for i in range(len(selected_classes)):
                        idx = (self.labels==selected_classes[i]).nonzero()[0]
                        select_instances = np.random.choice(idx, self.shots, False)
                        lam = np.random.beta(5, 1)

                        #if selected_classes[i] in [2,15]:
                        #    cls_idx[selected_classes[i]] = i
                        #lam = np.random.uniform(0.49,0.51)

                        if i<self.ways/2:
                            for j in range(self.shots):
                                    feat = self.feats[select_instances[j], :]
                                    att = self.atts[select_instances[j], :]  

                                    feat = feat.unsqueeze(0)
                                    att = att.unsqueeze(0)
                                    # print(feat.size())
                                    # print(att.size())
                                    if is_first:
                                            is_first=False
                                            select_feats = feat
                                            select_atts = att
                                    else:                   
                                            select_feats = torch.cat((select_feats, feat),0)
                                            select_atts = torch.cat((select_atts, att),0)
                                    select_labels[i*self.shots+j] = i
                        else:
                            for j in range(self.shots):
                                    feat = self.feats[select_instances[j], :]
                                    att = self.atts[select_instances[j], :]  

                                    feat = feat.unsqueeze(0)
                                    feat = lam*feat+(1-lam)*select_feats[int(i-self.ways/2)*self.shots+j]
                                    att = att.unsqueeze(0)
                                    att = lam*att+(1-lam)*select_atts[int(i-self.ways/2)*self.shots+j]
                                    # print(feat.size())
                                    # print(att.size())
                                    select_feats = torch.cat((select_feats, feat),0)                
                                    select_atts = torch.cat((select_atts, att),0)                
                                    select_labels[i*self.shots+j] = i - int(self.ways/2)

                noval_index = int(self.ways/2)*self.shots

                return select_feats[noval_index:], select_atts[noval_index:], select_labels[noval_index:]

        def __our_getitem__(self, index, netM, netAE):
                is_first = True
                select_feats = []
                select_atts = []
                select_labels = []
                select_labels = torch.LongTensor(self.ways*self.shots)
                selected_classes = np.random.choice(list(self.classes), self.ways, False) # AWA2和APY需要把这里改为True,CUN和SUN可以是False
                #selected_classes = self.classes
                #mixup
                cls_idx = {}


                for i in range(len(selected_classes)):
                        idx = (self.labels==selected_classes[i]).nonzero()[0]
                        select_instances = np.random.choice(idx, self.shots, False)
                        lam = np.random.beta(5, 1)

                        #if selected_classes[i] in [2,15]:
                        #    cls_idx[selected_classes[i]] = i
                        #lam = np.random.uniform(0.49,0.51)

                        if i<self.ways/2:
                            for j in range(self.shots):
                                    feat = self.feats[select_instances[j], :]
                                    att = self.atts[select_instances[j], :]

                                    feat = feat.unsqueeze(0)
                                    att = att.unsqueeze(0)
                                    # print(feat.size())
                                    # print(att.size())
                                    if is_first:
                                            is_first=False
                                            select_feats = feat
                                            select_atts = att
                                    else:
                                            select_feats = torch.cat((select_feats, feat),0)
                                            select_atts = torch.cat((select_atts, att),0)
                                    select_labels[i*self.shots+j] = i
                        else:
                            for j in range(self.shots):
                                    feat = self.feats[select_instances[j], :]
                                    att = self.atts[select_instances[j], :]

                                    feat = feat.unsqueeze(0)
                                    select_feat = select_feats[int(i-self.ways/2)*self.shots+j]
                                    select_feat = select_feat.unsqueeze(0)
                                    # feat = lam*feat+(1-lam)*select_feats[int(i-self.ways/2)*self.shots+j]
                                    att = att.unsqueeze(0)
                                    att_v = lam*att+(1-lam)*select_atts[int(i-self.ways/2)*self.shots+j]
                                    # netN
                                    # _, _, v_feat = netn(feat.cuda(), lam, select_feat.cuda())
                                    # SDGZSL
                                    x_mean, _, _, _ = netM(feat.cuda(), att.cuda())
                                    v_feat, h1, hs1, hn1 = netAE(x_mean)

                                    # print(feat.size())
                                    # print(att.size())
                                    select_feats = torch.cat((select_feats.cuda(), v_feat.cuda()),0)
                                    select_atts = torch.cat((select_atts, att_v),0)
                                    select_labels[i*self.shots+j] = i - int(self.ways/2)


                noval_index = int(self.ways/2)*self.shots

                return select_feats[noval_index:], select_atts[noval_index:], select_labels[noval_index:]

        def __train_newnet__(self, index, netn, neta, optimizer_netn, optimizer_neta):
                is_first = True
                select_feats = []
                select_atts = []
                select_labels = []
                select_labels = torch.LongTensor(self.ways * self.shots)
                selected_classes = np.random.choice(list(self.classes), self.ways, False)  # AWA2和APY需要把这里改为True,CUN和SUN可以是False
                # selected_classes = self.classes
                # mixup
                cls_idx = {}
                loss_visual = 0
                loss_att = 0
                feat_vec = []
                select_feat_vec = []
                att_vec = []
                select_att_vec = []
                v_att_vec = []
                lam = 0

                for i in range(len(selected_classes)):
                        idx = (self.labels == selected_classes[i]).nonzero()[0]
                        select_instances = np.random.choice(idx, self.shots, False)
                        lam = np.random.beta(5, 1)
                        if i < self.ways / 2:
                                for j in range(self.shots):
                                        feat = self.feats[select_instances[j], :]
                                        att = self.atts[select_instances[j], :]

                                        feat = feat.unsqueeze(0)
                                        att = att.unsqueeze(0)
                                        if is_first:
                                                is_first = False
                                                select_feats = feat
                                                select_atts = att
                                        else:
                                                select_feats = torch.cat((select_feats, feat), 0)
                                                select_atts = torch.cat((select_atts, att), 0)
                                        select_labels[i * self.shots + j] = i
                        else:
                                idx = (self.labels == selected_classes[i]).nonzero()[0]
                                select_instances = np.random.choice(idx, self.shots, False)
                                lam = np.random.beta(5, 1)

                                is_first = True

                                # feat_vec = []
                                # select_feat_vec = []
                                # att_vec = []
                                # select_att_vec = []
                                # v_att_vec = []

                                for j in range(self.shots):
                                        feat = self.feats[select_instances[j], :]
                                        att = self.atts[select_instances[j], :]

                                        feat = feat.unsqueeze(0)
                                        select_feat = select_feats[int(i - self.ways / 2) * self.shots + j].cuda()
                                        select_feat = select_feat.unsqueeze(0)

                                        att = att.unsqueeze(0)
                                        att_v = lam * att + (1 - lam) * select_atts[int(i - self.ways / 2) * self.shots + j]
                                        select_att = select_atts[int(i - self.ways / 2) * self.shots + j]
                                        select_att = select_att.unsqueeze(0)

                                        if is_first:
                                                is_first = False
                                                feat_vec = feat
                                                select_feat_vec = select_feat
                                                att_vec = att
                                                select_att_vec = select_att
                                                v_att_vec = att_v
                                        else:
                                                feat_vec = torch.cat((feat_vec, feat),0)
                                                select_feat_vec = torch.cat((select_feat_vec, select_feat), 0)
                                                att_vec = torch.cat((att_vec, att), 0)
                                                select_att_vec = torch.cat((select_att_vec, select_att), 0)
                                                v_att_vec = torch.cat((v_att_vec, att_v), 0)

                mse1 = torch.nn.MSELoss(reduction="mean")
                # Training netA
                for p in neta.parameters():
                        p.requires_grad_(True)
                for p in netn.parameters():
                        p.requires_grad_(False)
                neta.zero_grad()
                i_att = neta(feat_vec.cuda())
                j_att = neta(select_feat_vec.cuda())
                loss_att = mse1(i_att.cuda(), att_vec.cuda()) + mse1(j_att.cuda(), select_att_vec.cuda())
                loss_att.backward()
                optimizer_neta.step()

                # Training netN
                for p in neta.parameters():
                        p.requires_grad_(False)
                for p in netn.parameters():
                        p.requires_grad_(True)
                netn.zero_grad()
                i_feat, j_feat, v_feat = netn(feat_vec.cuda(), lam, select_feat_vec.cuda())
                v_att = neta(v_feat.cuda())
                loss_visual = mse1(i_feat.cuda(), feat_vec.cuda()) + mse1(j_feat.cuda(), select_feat_vec.cuda()) + mse1(
                        v_att.cuda(), v_att_vec.cuda())
                loss_visual.backward()
                optimizer_netn.step()

                loss = loss_visual + loss_att
                loss_visual_mse_vatt = mse1(v_att.cuda(), v_att_vec.cuda())
                return loss, loss_visual, loss_att, loss_visual_mse_vatt


        def __train_ae__(self, index, ae, neta, ae_optimizer, optimizer_neta, discriminator, dis_optimizer, relationNet,relationNet_optimizer,  ones, zeros):


                is_first = True
                select_feats = []
                select_atts = []
                select_labels = []
                select_labels = torch.LongTensor(self.ways * self.shots)
                selected_classes = np.random.choice(list(self.classes), self.ways, True)  # AWA2和APY需要把这里改为True,CUN和SUN可以是False
                # selected_classes = self.classes
                # mixup
                cls_idx = {}
                loss_visual = 0
                loss_xi = 0
                loss_xj = 0
                loss_att = 0
                feat_vec = []
                select_feat_vec = []
                att_vec = []
                select_att_vec = []
                v_att_vec = []
                lam = 0
                ntrain = self.feats.size()[0]
                args = Options().parse()

                for i in range(len(selected_classes)):
                        idx = (self.labels == selected_classes[i]).nonzero()[0]
                        print(idx)
                        select_instances = np.random.choice(idx, self.shots, False)
                        lam = np.random.beta(5, 1)
                        if i < self.ways / 2:
                                for j in range(self.shots):
                                        feat = self.feats[select_instances[j], :]
                                        att = self.atts[select_instances[j], :]

                                        labels = self.labels[select_instances[j]]

                                        feat = feat.unsqueeze(0)
                                        att = att.unsqueeze(0)
                                        if is_first:
                                                is_first = False
                                                select_feats = feat
                                                select_atts = att
                                        else:
                                                select_feats = torch.cat((select_feats, feat), 0)
                                                select_atts = torch.cat((select_atts, att), 0)
                                        select_labels[i * self.shots + j] = i
                        else:
                                idx = (self.labels == selected_classes[i]).nonzero()[0]
                                select_instances = np.random.choice(idx, self.shots, False)
                                lam = np.random.beta(5, 1)

                                is_first = True

                                # feat_vec = []
                                # select_feat_vec = []
                                # att_vec = []
                                # select_att_vec = []
                                # v_att_vec = []

                                for j in range(self.shots):
                                        feat = self.feats[select_instances[j], :]
                                        att = self.atts[select_instances[j], :]
                                        labels = self.labels[select_instances[j]]

                                        feat = feat.unsqueeze(0)
                                        select_feat = select_feats[int(i - self.ways / 2) * self.shots + j].cuda()
                                        select_feat = select_feat.unsqueeze(0)

                                        att = att.unsqueeze(0)
                                        att_v = lam * att + (1 - lam) * select_atts[int(i - self.ways / 2) * self.shots + j]
                                        select_att = select_atts[int(i - self.ways / 2) * self.shots + j]
                                        select_att = select_att.unsqueeze(0)

                                        if is_first:
                                                is_first = False
                                                feat_vec = feat
                                                select_feat_vec = select_feat
                                                att_vec = att
                                                select_att_vec = select_att
                                                v_att_vec = att_v
                                        else:
                                                feat_vec = torch.cat((feat_vec, feat),0)
                                                select_feat_vec = torch.cat((select_feat_vec, select_feat), 0)
                                                att_vec = torch.cat((att_vec, att), 0)
                                                select_att_vec = torch.cat((select_att_vec, select_att), 0)
                                                v_att_vec = torch.cat((v_att_vec, att_v), 0)

                labels_tensor = torch.tensor(labels).long()
                mapped_labels_tensor = map_label(labels_tensor, self.seen_classes).unsqueeze(0)

                mse1 = torch.nn.MSELoss(reduction="mean")
                # Training netA
                for p in neta.parameters():
                        p.requires_grad_(True)
                for p in ae.parameters():
                        p.requires_grad_(False)
                neta.zero_grad()
                i_att = neta(feat_vec.cuda())
                j_att = neta(select_feat_vec.cuda())
                loss_att = mse1(i_att.cuda(), att_vec.cuda()) + mse1(j_att.cuda(), select_att_vec.cuda())
                loss_att.backward()
                optimizer_neta.step()

                # Training netN
                # for p in neta.parameters():
                #         p.requires_grad_(False)
                # for p in netn.parameters():
                #         p.requires_grad_(True)
                # netn.zero_grad()
                # i_feat, j_feat, v_feat = netn(feat_vec.cuda(), lam, select_feat_vec.cuda())
                # v_att = neta(v_feat.cuda())
                # loss_visual = mse1(i_feat.cuda(), feat_vec.cuda()) + mse1(j_feat.cuda(), select_feat_vec.cuda()) + mse1(
                #         v_att.cuda(), v_att_vec.cuda())
                # loss_visual.backward()
                # optimizer_netn.step()
                start_step = 0
                args.niter = int(ntrain/args.batchSize) * args.gen_nepoch
                iters = math.ceil(ntrain/args.batchSize)
                beta = 0.01
                coin = 0
                gamma = 0
                for it in range(start_step, args.niter+1):
                        if it % iters == 0:
                                beta = min(args.kl_warmup*(it/iters), 1)
                                gamma = min(args.tc_warmup*(it/iters), 1)
                        # train_att = np.array([self.atts[i,:] for i in self.labels])
                        # train_att = torch.from_numpy((train_att.astype('float32')))
                        # train_feat = torch.from_numpy(self.feats)

                        # self.labels_tensor = torch.tensor(self.labels).long()
                        mapped_labels_tensor_unique = mapped_labels_tensor.unique()
                        mapped_labels_tensor_unique_num = mapped_labels_tensor_unique.shape[0]
                        self.dim_s = self.atts.shape[1]
                        sample_att = torch.zeros((mapped_labels_tensor_unique_num,self.dim_s)).cuda()
                        for iter_sample_att in range(mapped_labels_tensor_unique_num):
                                sample_att[iter_sample_att] = self.atts[mapped_labels_tensor_unique[iter_sample_att]]
                        # sample_att = torch.from_numpy(np.array([self.atts[i,:] for i in self.labels_tensor.unique()])).cuda()
                        sample_att_n = mapped_labels_tensor_unique_num
                        sample_labels = mapped_labels_tensor_unique
                        re_batch_labels = []
                        for label in mapped_labels_tensor:
                                ind = torch.argwhere(sample_labels == label)
                                re_batch_labels.append(ind[0][0])
                        re_batch_labels = torch.LongTensor(re_batch_labels)
                        one_hot_labels = torch.zeros(args.batchSize, sample_att_n).scatter_(1, re_batch_labels.view(-1,1), 1).cuda()
                        x_i, zi, zi_s, zi_ns = ae(feat_vec.cuda())
                        x_j, zj, zj_s, zj_ns = ae(select_feat_vec.cuda())
                        # xi
                        print(feat_vec.shape)
                        print(zi_s.shape)
                        relations = relationNet(zi_s, sample_att)
                        print(relations.shape)
                        relations = relations.view(-1, mapped_labels_tensor_unique.cpu().shape[0])
                        print(relations.shape)
                        print(one_hot_labels.shape)
                        p_loss = args.ga * mse1(relations, one_hot_labels)
                        rec = mse1(x_i, feat_vec)
                        if coin > 0:
                                s_score = discriminator(zi)
                                tc_loss = args.beta * gamma *((s_score[:, :1] - s_score[:, 1:]).mean())
                                loss_xi = p_loss + rec + tc_loss
                                coin -= 1
                        else:
                                s, n = permute_dims(zi_s, zi_ns)
                                b = torch.cat((s, n), 1).detach()
                                s_score = discriminator(zi)
                                n_score = discriminator(b)
                                tc_loss = args.dis * (F.cross_entropy(s_score, zeros) + F.cross_entropy(n_score, ones))
                                dis_optimizer.zero_grad()
                                tc_loss.backward(retain_graph=True)
                                dis_optimizer.step()
                                loss_xi = p_loss +rec
                                coin += args.dis_step
                        relationNet_optimizer.zero_grad()
                        ae_optimizer.zero_grad()
                        # loss_xi.backward()
                        relationNet_optimizer.step()
                        ae_optimizer.step()

                        # xj
                        relations = relationNet(zj_s, sample_att)
                        relations = relations.view(-1, self.labels.unique().cpu().shape[0])
                        p_loss = args.ga * mse1(relations, one_hot_labels)
                        rec = mse1(x_j, select_feat_vec)
                        if coin > 0:
                                s_score = discriminator(zj)
                                tc_loss = args.beta * gamma * ((s_score[:, :1] - s_score[:, 1:]).mean())
                                loss_xj = p_loss + rec + tc_loss
                                coin -= 1
                        else:
                                s, n = permute_dims(zj_s, zj_ns)
                                b = torch.cat((s, n), 1).detach()
                                s_score = discriminator(zj)
                                n_score = discriminator(b)
                                tc_loss = args.dis * (F.cross_entropy(s_score, zeros) + F.cross_entropy(n_score, ones))
                                dis_optimizer.zero_grad()
                                tc_loss.backwar(retain_graph=True)
                                dis_optimizer.step()
                                loss_xj = p_loss + rec
                                coin += args.dis_step
                        loss_visual = loss_xi + loss_xj
                        relationNet_optimizer.zero_grad()
                        ae_optimizer.zero_grad()
                        loss_visual.backward()
                        relationNet_optimizer.step()
                        ae_optimizer.step()
                loss = loss_visual + loss_att
                return loss, loss_visual, loss_att


        def __len__(self):
                return self.__size

