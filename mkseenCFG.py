#l/bin/env/python
#-*-coding:utf-8-*-

"""
    create yaml for cfg(configuration)
"""
import argparse
import yaml

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

cfg = {
        'dataset' : {
            'name':'APY',
            'num_classes':32,
            'semantic_type':'att',
            'semantic_dim':64,
            },
        'model_hyp':{
            'lr': 7.07e-6, # initial learning rate
            'momentum':0.97, # SGD momentum
            'weight_decay': 4e-2, # optimizer weight decay
            'epochs':100,
            'start_epoch':0,
            'img_feats':2048,    # imgf features dimension
            'att_feats':64,        # att features dimension
            'distributed':False,
            'workers':16,        # workers to load data
            'dropout':0.5,      # dropout rate of features while training
            'accType':'allCls',
            'ClassifierType':'Cos',     # ReLUCos or GauK(gaussianKernel)
            'Wnorm':True,
            'random_seend':0,
            },
        'dataset_hyp':{
            'img_resize':[224,224],
            'batch_size':64,
            'imgfType':'customed',
            }

    }

if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument('--yamlFile',default='./chk/seenExpert.yaml',type=str,help='yaml file')
    parse.add_argument('--random_seed',type=int,default=520)
    parse.add_argument('--lr',type=float,default=5e-4)
    parse.add_argument('--weightDecay',type=float,default=9e-7)
    parse.add_argument('--imgType',default='CustomedExtracted',type=str,help='image features from customed resnet or xlsa')
    parse.add_argument('--datasetName',default='APY',type=str,help='dataset Name')
    opt = parse.parse_args()
    print(opt)
    cfg['model_hyp']['random_seed'] = opt.random_seed
    cfg['model_hyp']['lr'] = opt.lr
    cfg['model_hyp']['weight_decay'] = opt.weightDecay
    cfg['dataset_hyp']['imgfType'] = opt.imgType.lower()
    cfg['dataset']['name'] = opt.datasetName

    with open(opt.yamlFile,'w') as f:
        yaml.dump(cfg,f,default_flow_style=False)
        print("renew yaml")

