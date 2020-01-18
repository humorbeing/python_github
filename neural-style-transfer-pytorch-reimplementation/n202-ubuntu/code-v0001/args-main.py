from argparse import Namespace
from trainer import train

work_root = '../../../__SSSSTTTTOOOORRRREEEE/neural-style/'


args = Namespace()
args.epochs = 2
args.batch_size = 4
args.dataset = '../../../__SSSSTTTTOOOORRRREEEE/coco-dataset'
# args.dataset = '../../../__SSSSTTTTOOOORRRREEEE/coco-dataset/'
args.save_model_dir = work_root + 'saved-model-here/'
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

args.is_quickrun = True
style_images_root = '../../style-images/style-images-here/'
numbers = [
    '01','02','03','04','05',
    '06','07','08','09','10'
]

for num in numbers:
    args.style_name = num
    args.style_image = style_images_root + num + '.jpg'
    train(args)