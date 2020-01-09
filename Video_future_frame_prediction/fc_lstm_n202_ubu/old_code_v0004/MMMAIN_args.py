from argparse import Namespace
from runner import runner

path = '../../__SSSSTTTTOOOORRRREEEE/Data_save_here/'

def get_args():
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
    args.loss_function = 'bce'  # 'mse' / 'bce'
    # NOTE: 'bce' must coupled with sigmoid and is_standardization=False
    args.hidden = 2048
    args.mode = 'pred'  # 'recon' / 'pred' / 'both'
    args.zero_input = False
    args.seed = 6
    args.recon_loss_lambda = 0.8
    args.optimizer = 'rmsprop'  # 'rmsprop' / 'adam'
    args.learning_rate = 0.0001

    return args

#///////////////////////////////////////////////////////

args = get_args()
args.is_cuda = False
args.is_quickrun = True
args.is_save = False
args.mode = 'both'
runner(args, path)




