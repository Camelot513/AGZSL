import numpy as np
import torch
import torch.nn as nn
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
        self.decoder_sigmoid = nn.Sigmoid()

        #attReg
        self.attReg_linear1 = nn.Linear(self.vz, self.hz)
        self.attReg_lrelu1 = nn.LeakyReLU(0.2, True)
        self.attReg_linear2 = nn.Linear(self.hz, self.att_size)
        self.attReg_sigmoid = nn.Sigmoid()

    def forward(self, feat, lam, select_feats):
        #use encoder
        encoder_h1 = self.encoder_lrelu1(self.encoder_linear1(feat))
        zi = self.encoder_linear2(encoder_h1)
        encoder_h2 = self.encoder_lrelu1(self.encoder_linear1(select_feats))
        zj = self.encoder_linear2(encoder_h2)
        zv = lam*zi + (1-lam)*zj

        #use decoder
        decoder_h1 = self.decoder_lrelu1(self.decoder_linear1(zi))
        decoder_h2 = self.decoder_linear2(decoder_h1)
        x_i = self.decoder_sigmoid(decoder_h2)
        decoder_h3 = self.decoder_lrelu1(self.decoder_linear1(zj))
        decoder_h4 = self.decoder_linear2(decoder_h3)
        x_j = self.decoder_sigmoid(decoder_h4)
        decoder_h5 = self.decoder_lrelu1(self.decoder_linear1(zv))
        decoder_h6 = self.decoder_linear2(decoder_h5)
        x_v = self.decoder_sigmoid(decoder_h6)

        #use attReg
        attReg_h1 = self.attReg_lrelu1(self.attReg_linear1(x_i))
        attReg_h2 = self.attReg_linear2(attReg_h1)
        a_i = self.attReg_sigmoid(attReg_h2)
        attReg_h3 = self.attReg_lrelu1(self.attReg_linear1(x_j))
        attReg_h4 = self.attReg_linear2(attReg_h3)
        a_j = self.attReg_sigmoid(attReg_h4)
        attReg_h5 = self.attReg_lrelu1(self.attReg_linear1(x_v))
        attReg_h6 = self.attReg_linear2(attReg_h5)
        a_v = self.attReg_sigmoid(attReg_h6)

        return a_i, a_j, a_v, x_i, x_j, x_v


