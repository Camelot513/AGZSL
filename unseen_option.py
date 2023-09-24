"""
This code is from Kai Li "kailigo". The website is https://github.com/kailigo/cvcZSL.
"""
import argparse
import os

class Options():
    def __init__(self):
        # Training settings

        parser = argparse.ArgumentParser(description='Tank Shot')

        parser.add_argument('--dataset', default='APY', type=str,
                            help='dataset to be processed')
        
        # parser.add_argument('--batchSize', default=25,type=int,
        #                     help='Batch Size')
        # parser.add_argument('--lr', default=1e-3, type=float,
        #                     help='learning rate')
        parser.add_argument('--step_size', default=100, type=int,
                            help='decay step')
        parser.add_argument('--gamma', default=0.5, type=float,
                            help='decay rate')
        parser.add_argument('--num_epochs', default=200, type=int,
                            help='epoch number')                
        parser.add_argument('--nthreads', default=8,type=int,
                            help='threads num to load data')

        parser.add_argument('--ways', default=16,type=int,
                            help='number of class for one test')
        # CUB and SUN: 20
        # AWA2 and APY: 16
        parser.add_argument('--shots', default=4,type=int,
                            help='number of pictures of each class to support')

        parser.add_argument('--lr', default=1e-5,type=float,
                            help='learning rate')
            
        parser.add_argument('--weight_model', default='weightnet',type=str,
                            help='weight model name after finetuning')

        parser.add_argument('--hidden_dim', default=1600,type=int,
                            help='hidden dimension')

        parser.add_argument('--opt_decay', default=1e-3,type=float,
                            help='decay rate for optimizer')
        
        parser.add_argument("--log_to_file", type=bool, default=True)
        parser.add_argument("--log_file", type=str, default='temp.log')
    
        parser.add_argument("--model_file", type=str, default='model.pt')
        parser.add_argument('--device', default='cuda:1', type=str, help='cuda:0,cuda:1,cuda:2,cpu')

        #基于net-SRWGAN新增
        parser.add_argument('--nz', type=int, default=2048, help='size of the semantic representation * 2')
        parser.add_argument('--ngh', type=int, default=2048, help='default value')
        parser.add_argument('--attSize', type=int, default=4097, help='default value')
        parser.add_argument('--resSize', type=int, default=2048, help='default value')
        parser.add_argument('--beta1', default=0.5, type=float, help='default value')
        parser.add_argument('--inputSize', type=int, default=4097, help='default value')

        #for our new_network
        parser.add_argument('--vz', type=int, default=2048, help='dataset visual_dim')
        parser.add_argument('--hz', type=int, default=1024, help='size of the visual representation * 2') # 原1024
        parser.add_argument('--iz', type=int, default=2048, help='dataset visual_dim')
        parser.add_argument('--pre_epochs', type=int, default=160, help='pre_net_train,same as num_epochs')
        parser.add_argument('--att_size', type=int, default=64, help='dataset semantic size')
        # SUN:102, CUB:312, AWA2:85, APY:64

        # develop network
        parser.add_argument('--S_dim', type=int, default=1024)
        parser.add_argument('--NS_dim', type=int, default=1024)
        parser.add_argument('--C_dim', type=int, default=312, help='dataset semantic size')
        parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
        parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
        parser.add_argument('--kl_warmup', type=float, default=0.01, help='kl warm-up for VAE')
        parser.add_argument('--tc_warmup', type=float, default=0.001, help='tc warm-up')
        parser.add_argument('--ga', type=float, default=15, help='relationNet weight')
        parser.add_argument('--beta', type=float, default=1, help='tc weight')
        parser.add_argument('--dis', type=float, default=3, help='Discriminator weight')
        parser.add_argument('--dis_step', type=float, default=2, help='Discriminator update interval')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()