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
args.is_quickrun = False
# default combos
args.is_standardization = False
args.last_activation = 'sigmoid'  # 'tanh' / 'sigmoid' / 'non'
args.loss_function = 'mse'  # 'mse' / 'bce'
# NOTE: 'bce' must coupled with sigmoid and is_standardization=False
args.hidden = 2048
args.mode = 'recon'  # 'recon' / 'pred' / 'both'
args.zero_input = True
args.seed = 6
args.recon_loss_lambda = 0.8

#///////////////////////////////////////////////////////
a = Namespace(**vars(args))
b = Namespace(**vars(args))
c = Namespace(**vars(args))

a.is_cuda = False
a.is_quickrun = True
a.is_save = False
a.zero_input = True
#0 s m
runner(a, path)
# 0 s b
a.loss_function = 'bce'
runner(a, path)
# 0 n m
a.last_activiation = 'non'
a.loss_function = 'mse'
runner(a, path)
#1 t m
a.is_standardization = True
a.last_activation = 'tanh'  # 'tanh' / 'sigmoid' / 'non'
a.loss_function = 'mse'  # 'mse' / 'bce'
runner(a, path)
# 1 n m
a.is_standardization = True
a.last_activation = 'non'  # 'tanh' / 'sigmoid' / 'non'
a.loss_function = 'mse'  # 'mse' / 'bce'
runner(a, path)


a.zero_input = False
a.is_standardization = False
a.last_activation = 'sigmoid'  # 'tanh' / 'sigmoid' / 'non'
a.loss_function = 'mse'
#0 s m
runner(a, path)
# 0 s b
a.loss_function = 'bce'
runner(a, path)
# 0 n m
a.last_activiation = 'non'
a.loss_function = 'mse'
runner(a, path)
#1 t m
a.is_standardization = True
a.last_activation = 'tanh'  # 'tanh' / 'sigmoid' / 'non'
a.loss_function = 'mse'  # 'mse' / 'bce'
runner(a, path)
# 1 n m
a.is_standardization = True
a.last_activation = 'non'  # 'tanh' / 'sigmoid' / 'non'
a.loss_function = 'mse'  # 'mse' / 'bce'
runner(a, path)


##////////////////////////////////////////////
a.mode = 'pred'  # 'recon' / 'pred' / 'both'
a.zero_input = True
#0 s m
runner(a, path)
# 0 s b
a.loss_function = 'bce'
runner(a, path)
# 0 n m
a.last_activiation = 'non'
a.loss_function = 'mse'
runner(a, path)
#1 t m
a.is_standardization = True
a.last_activation = 'tanh'  # 'tanh' / 'sigmoid' / 'non'
a.loss_function = 'mse'  # 'mse' / 'bce'
runner(a, path)
# 1 n m
a.is_standardization = True
a.last_activation = 'non'  # 'tanh' / 'sigmoid' / 'non'
a.loss_function = 'mse'  # 'mse' / 'bce'
runner(a, path)


a.zero_input = False
a.is_standardization = False
a.last_activation = 'sigmoid'  # 'tanh' / 'sigmoid' / 'non'
a.loss_function = 'mse'
#0 s m
runner(a, path)
# 0 s b
a.loss_function = 'bce'
runner(a, path)
# 0 n m
a.last_activiation = 'non'
a.loss_function = 'mse'
runner(a, path)
#1 t m
a.is_standardization = True
a.last_activation = 'tanh'  # 'tanh' / 'sigmoid' / 'non'
a.loss_function = 'mse'  # 'mse' / 'bce'
runner(a, path)
# 1 n m
a.is_standardization = True
a.last_activation = 'non'  # 'tanh' / 'sigmoid' / 'non'
a.loss_function = 'mse'  # 'mse' / 'bce'
runner(a, path)


## //////////////////////////////////////////////////////////////

a.mode = 'both'  # 'recon' / 'pred' / 'both'
a.zero_input = True
#0 s m
runner(a, path)
# 0 s b
a.loss_function = 'bce'
runner(a, path)
# 0 n m
a.last_activiation = 'non'
a.loss_function = 'mse'
runner(a, path)
#1 t m
a.is_standardization = True
a.last_activation = 'tanh'  # 'tanh' / 'sigmoid' / 'non'
a.loss_function = 'mse'  # 'mse' / 'bce'
runner(a, path)
# 1 n m
a.is_standardization = True
a.last_activation = 'non'  # 'tanh' / 'sigmoid' / 'non'
a.loss_function = 'mse'  # 'mse' / 'bce'
runner(a, path)


a.zero_input = False
a.is_standardization = False
a.last_activation = 'sigmoid'  # 'tanh' / 'sigmoid' / 'non'
a.loss_function = 'mse'
#0 s m
runner(a, path)
# 0 s b
a.loss_function = 'bce'
runner(a, path)
# 0 n m
a.last_activiation = 'non'
a.loss_function = 'mse'
runner(a, path)
#1 t m
a.is_standardization = True
a.last_activation = 'tanh'  # 'tanh' / 'sigmoid' / 'non'
a.loss_function = 'mse'  # 'mse' / 'bce'
runner(a, path)
# 1 n m
a.is_standardization = True
a.last_activation = 'non'  # 'tanh' / 'sigmoid' / 'non'
a.loss_function = 'mse'  # 'mse' / 'bce'
runner(a, path)