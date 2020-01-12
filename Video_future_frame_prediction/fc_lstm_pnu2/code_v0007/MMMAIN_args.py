from argparse import Namespace
from runner import runner

path = '../../../__SSSSTTTTOOOORRRREEEE/Data_save_here/'

def get_args():
    args = Namespace()
    args.batch_size = 100
    args.epoch = 200
    args.model = 'lstm_copy'  # 'lstm_copy' /
    args.is_cuda = True
    args.data_path = path
    args.log_path = './'
    args.save_path = path+'fc_lstm_model_save/model_save/'
    args.is_save = True
    args.is_quickrun = False
    # default combos
    args.last_activation = 'non'  # '100s' / 'sigmoid' / 'non'
    args.loss_function = 'mse'  # 'mse' / 'bce'
    args.hidden = 2048
    args.mode = 'both'  # 'recon' / 'pred' / 'both'
    args.zero_input = False
    args.seed = 6
    args.recon_loss_lambda = 0.8
    args.optimizer = 'adam'  # 'rmsprop' / 'adam'
    args.learning_rate = 0.001
    args.is_init = True
    args.is_lr_scheduler = True
    args.lr_scheduler_inteval = 30
    args.lr_scheduler_gamma = 0.5
    args.gradiant_clip = 0.25
    args.is_show = True
    return args

#///////////////////////////////////////////////////////

args = get_args()
args.batch_size = 50
# args.is_cuda = False
# args.is_quickrun = True
# args.is_save = True
# args.mode = 'recon'
# args.optimizer = 'adam'
args.last_activation = '100s'  # '100s' / 'sigmoid' / 'non'
args.loss_function = 'bce'  # 'mse' / 'bce'
runner(args)
# for i in vars(args):
#     print('ARGS >>> ' + i + ' :{}'.format(vars(args)[i]))



