from argparse import Namespace
from runner import runner

path = '../../__SSSSTTTTOOOORRRREEEE/Data_save_here/'

args = Namespace()

args.batch_size = 100
args.epoch = 200
args.model = 'ED_R_01'  # 'ED_R_01' /
args.is_cuda = True
args.work_path = './'
args.save_path = path+'fc_lstm_model_save/model_save/'
args.is_save = True
# default combos
args.is_standardization = True
args.last_activation = 'tanh'  # 'tanh' / 'sigmoid' / 'non'
args.loss_function = 'mse'  # 'mse' / 'bce'

args.mode = 'recon'  # 'recon' / 'pred' / 'both'
args.zero_input = True
args.seed = 6

#///////////////////////////////////////////////////////
a = Namespace(**vars(args))
b = Namespace(**vars(args))
c = Namespace(**vars(args))

a.is_cuda = False


runner(a, path)