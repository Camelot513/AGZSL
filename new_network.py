import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#         m.bias.data.fill_(0)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            init.normal_(m.weight.data, mean=0, std=0.02)
            if m.bias is not None:
                init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
class new_network(nn.Module):
    def __init__(self, args):
        super(new_network, self).__init__()
        self.iz = args.iz
        self.hz = args.hz
        self.vz = args.vz
        self.att_size = args.att_size
        # Encoder
        self.encoder_linear1 = nn.Linear(self.iz, self.hz)
        self.encoder_lrelu1 = nn.LeakyReLU(0.2, True)
        self.encoder_linear2 = nn.Linear(self.hz, self.iz)

        #Decoder
        self.decoder_linear1 = nn.Linear(self.iz, self.hz)
        self.decoder_lrelu1 = nn.LeakyReLU(0.2, True)
        self.decoder_linear2 = nn.Linear(self.hz, self.vz)
        # self.decoder_sigmoid = nn.Sigmoid()

        # #attReg
        # self.attReg_linear1 = nn.Linear(self.vz, self.hz)
        # self.attReg_lrelu1 = nn.LeakyReLU(0.2, True)
        # self.attReg_linear2 = nn.Linear(self.hz, self.att_size)
        # self.apply(weights_init)
        initialize_weights(self)

    def forward(self, feat, lam, select_feat):
        #use encoder
        encoder_h1 = self.encoder_lrelu1(self.encoder_linear1(feat))
        zi = self.encoder_linear2(encoder_h1)
        encoder_h2 = self.encoder_lrelu1(self.encoder_linear1(select_feat))
        zj = self.encoder_linear2(encoder_h2)
        zv = lam*zi + (1-lam)*zj

        #use decoder
        decoder_h1 = self.decoder_lrelu1(self.decoder_linear1(zi))
        x_i = self.decoder_linear2(decoder_h1)
        # x_i = self.decoder_sigmoid(decoder_h2)
        decoder_h3 = self.decoder_lrelu1(self.decoder_linear1(zj))
        x_j = self.decoder_linear2(decoder_h3)
        # x_j = self.decoder_sigmoid(decoder_h4)
        decoder_h5 = self.decoder_lrelu1(self.decoder_linear1(zv))
        x_v = self.decoder_linear2(decoder_h5)
        # x_v = self.decoder_sigmoid(decoder_h6)

        # #use attReg
        # attReg_h1 = self.attReg_lrelu1(self.attReg_linear1(x_i))
        # a_i = self.attReg_linear2(attReg_h1)
        # a_i = a_i / a_i.pow(2).sum(1).sqrt().unsqueeze(1).expand(a_i.size(0), a_i.size(1))
        # attReg_h2 = self.attReg_lrelu1(self.attReg_linear1(x_j))
        # a_j = self.attReg_linear2(attReg_h2)
        # a_j = a_j / a_j.pow(2).sum(1).sqrt().unsqueeze(1).expand(a_j.size(0), a_j.size(1))
        # attReg_h3 = self.attReg_lrelu1(self.attReg_linear1(x_v))
        # a_v = self.attReg_linear2(attReg_h3)
        # a_v = a_v / a_v.pow(2).sum(1).sqrt().unsqueeze(1).expand(a_v.size(0), a_v.size(1))
        x_i = F.normalize(x_i,p=2,dim=1,eps=1e-12)
        x_j = F.normalize(x_j,p=2,dim=1,eps=1e-12)
        x_v = F.normalize(x_v,p=2,dim=1,eps=1e-12)

        return x_i, x_j, x_v

class attNetwork(nn.Module):
    def __init__(self, args):
        super(attNetwork, self).__init__()
        self.iz = args.iz
        self.hz = args.hz
        self.vz = args.vz
        self.att_size = args.att_size
        # attReg
        self.attReg_linear1 = nn.Linear(self.vz, self.hz)
        self.attReg_lrelu1 = nn.LeakyReLU(0.2, True)
        self.attReg_linear2 = nn.Linear(self.hz, self.att_size)
        initialize_weights(self)

    def forward(self, feat):
        # use attReg
        attReg_h1 = self.attReg_lrelu1(self.attReg_linear1(feat))
        att_pred = self.attReg_linear2(attReg_h1)
        att_pred = att_pred / att_pred.pow(2).sum(1).sqrt().unsqueeze(1).expand(att_pred.size(0), att_pred.size(1))
        return att_pred

class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        self.iz = args.iz
        self.hz = args.hz
        self.vz = args.vz
        self.att_size = args.att_size
        # Encoder
        self.encoder_linear1 = nn.Linear(self.iz, self.hz)
        self.encoder_lrelu1 = nn.LeakyReLU(0.2, True)
        self.encoder_linear2 = nn.Linear(self.hz, self.iz)

        #Decoder
        self.decoder_linear1 = nn.Linear(self.iz, self.hz)
        self.decoder_lrelu1 = nn.LeakyReLU(0.2, True)
        self.decoder_linear2 = nn.Linear(self.hz, self.vz)
        # self.decoder_sigmoid = nn.Sigmoid()

        # #attReg
        # self.attReg_linear1 = nn.Linear(self.vz, self.hz)
        # self.attReg_lrelu1 = nn.LeakyReLU(0.2, True)
        # self.attReg_linear2 = nn.Linear(self.hz, self.att_size)
        # self.apply(weights_init)
        initialize_weights(self)

    def forward(self, feat, lam, select_feat):
        #use encoder
        encoder_h1 = self.encoder_lrelu1(self.encoder_linear1(feat))
        zi = self.encoder_linear2(encoder_h1)
        zi_s = zi[:, :self.args.S_dim]
        zi_ns = zi[:, self.args.S_dim:]
        encoder_h2 = self.encoder_lrelu1(self.encoder_linear1(select_feat))
        zj = self.encoder_linear2(encoder_h2)
        zj_s = zj[:, :self.args.S_dim]
        zj_ns = zj[:, self.args.S_dim:]
        zv = lam * zi + (1 - lam) * zj
        zv_s = zv[:, :self.args.S_dim]
        zv_ns = zv[:, self.args.S_dim:]

        # use decoder
        decoder_h1 = self.decoder_lrelu1(self.decoder_linear1(zi))
        x_i = self.decoder_linear2(decoder_h1)
        decoder_h3 = self.decoder_lrelu1(self.decoder_linear1(zj))
        x_j = self.decoder_linear2(decoder_h3)
        decoder_h5 = self.decoder_lrelu1(self.decoder_linear1(zv))
        x_v = self.decoder_linear2(decoder_h5)

        # #use attReg
        # attReg_h1 = self.attReg_lrelu1(self.attReg_linear1(x_i))
        # a_i = self.attReg_linear2(attReg_h1)
        # a_i = a_i / a_i.pow(2).sum(1).sqrt().unsqueeze(1).expand(a_i.size(0), a_i.size(1))
        # attReg_h2 = self.attReg_lrelu1(self.attReg_linear1(x_j))
        # a_j = self.attReg_linear2(attReg_h2)
        # a_j = a_j / a_j.pow(2).sum(1).sqrt().unsqueeze(1).expand(a_j.size(0), a_j.size(1))
        # attReg_h3 = self.attReg_lrelu1(self.attReg_linear1(x_v))
        # a_v = self.attReg_linear2(attReg_h3)
        # a_v = a_v / a_v.pow(2).sum(1).sqrt().unsqueeze(1).expand(a_v.size(0), a_v.size(1))

        x_i = F.normalize(x_i,p=2,dim=1,eps=1e-12)
        x_j = F.normalize(x_j,p=2,dim=1,eps=1e-12)
        x_v = F.normalize(x_v,p=2,dim=1,eps=1e-12)

        return x_i, zi, zi_s, zi_ns, x_j, zj, zj_s, zj_ns, x_v, zv, zv_s, zv_ns
class RelationNet(nn.Module):
    def __init__(self, args):
        super(RelationNet, self).__init__()
        self.fc1 = nn.Linear(args.C_dim + args.S_dim, 2048)
        self.fc2 = nn.Linear(2048, 1)

    def forward(self, s, c):

        c_ext = c.unsqueeze(0).repeat(s.shape[0], 1, 1)
        cls_num = c_ext.shape[1]

        s_ext = torch.transpose(s.unsqueeze(0).repeat(cls_num, 1, 1), 0, 1)
        relation_pairs = torch.cat((s_ext, c_ext), 2).view(-1, c.shape[1] + s.shape[1])
        relation = nn.ReLU()(self.fc1(relation_pairs))
        relation = nn.Sigmoid()(self.fc2(relation))
        return relation


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(args.S_dim*2, 2)

    def forward(self, s):
        score = self.fc1(s)
        return nn.Sigmoid()(score)
