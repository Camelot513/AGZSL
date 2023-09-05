"""
Part of this code is from Kai Li "kailigo". The website is https://github.com/kailigo/cvcZSL.
"""
from scipy import io
import numpy as np
import torch
from pdb import set_trace as breakpoint
import torch.utils.data as data
from sklearn.metrics import mean_squared_error



class data_loader_virtualCls(data.Dataset):
        def __init__(self, feats, atts, labels,  ways=16, shots=4):
                self.ways = ways*2                
                self.shots = shots    

                self.feats = torch.tensor(feats).float()
                self.atts = torch.tensor(atts).float()
                self.labels = labels
                self.classes = np.unique(labels)
                
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

        def __our_getitem__(self, index, netn):
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
                                    # select_att = select_atts[int(i-self.ways/2)*self.shots+j]
                                    # select_att = select_att.unsqueeze(0)
                                    # our
                                    # feat = ournet(lam,feat,select_feats)
                                    #netG
                                    # feat, G_noise1, G_noise2, G_noise3 = ournet(feat.cuda(), lam, select_feat)
                                    # att = att.unsqueeze(0)
                                    # att = lam * att + (1 - lam) * select_atts[int(i - self.ways / 2) * self.shots + j]

                                    # netN
                                    _, _, v_feat = netn(feat.cuda(), lam, select_feat.cuda())
                                    # netA
                                    # i_att, j_att, v_att = neta(feat.cuda(), select_feat.cuda(), v_feat.cuda())

                                    # att = lam*att+(1-lam)select_atts
                                    # or
                                    # att = ournet(lam,att,select_att)
                                    # feat = lam*feat+(1-lam)select_feat
                                    # our loss -> net

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


        def __len__(self):
                return self.__size

