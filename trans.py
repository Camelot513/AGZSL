import scipy.io
import pickle

# 加载.mat文件
matdata = scipy.io.loadmat('C:\someproject\ZSLearning\AGZSL-main\dataset\\finetune\CUB1\\res101.mat')

# 将数据保存为.pkl文件
with open('res101.pkl', 'wb') as f:
    pickle.dump(matdata, f)
