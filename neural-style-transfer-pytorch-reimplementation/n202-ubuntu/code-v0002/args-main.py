from argparse import Namespace
from trainer import train

work_root = '../../../__SSSSTTTTOOOORRRREEEE/neural-style/'

def get_args():
    args = Namespace()
    args.epochs = 2000
    args.batch_size = 8
    args.dataset = '../../../__SSSSTTTTOOOORRRREEEE/neural-style/family image outter folder'
    # args.dataset = '../../../__SSSSTTTTOOOORRRREEEE/coco-dataset'
    args.save_model_dir = work_root + 'saved-family-model-here/'
    args.style_image = '../../style-images/s.jpg'
    args.checkpoint_model_dir = ''
    args.image_size = 256
    args.style_size = None
    args.is_cuda = True
    args.seed = 42
    args.content_weight = 1e5
    args.style_weight = 1e10
    args.lr = 1e-3
    args.log_interval = 500
    args.checkpoint_interval = 2000
    args.style_name = 'mona'
    args.is_quickrun = False
    return args

args = get_args()
# args.is_quickrun = True
style_images_root = '../../style-images/style-images-here/'
numbers = [
    '01','02','03','04',
    '05','06','07','08'
]
args.style_name = '04'
args.style_image = style_images_root + args.style_name + '.jpg'
# args.style_image = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/s/s.jpg'
train(args)
# for num in numbers:
#     args.style_name = num
#     args.style_image = style_images_root + num + '.jpg'
#     train(args)