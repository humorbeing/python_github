### pytroch reimplementation
- copying code from pytorch examples
- [https://arxiv.org/abs/1603.08155](https://arxiv.org/abs/1603.08155)
### interesting part
- batch size 4
- epoch 2
- input 255
- loss weight 1e10
- generator, big pic and conv

### running
- training models
    - [pnu2] 1-4 training
    - [pnumy] 5-8 training
### code-v0001
- train on coco train2014 set
- trained model is saved and used later
### code-v0002
- only train on 8 family images
- try overfit it
### code-v0003
- gonna try only optimizer method
    - Not any more: [link](https://github.com/rrmina/neural-style-pytorch/blob/master/neural_style.ipynb)
        - bad implement
    - [link](https://nextjournal.com/gkoehler/pytorch-neural-style-transfer)
- folder names now have spaces
- this is from paper [https://arxiv.org/abs/1508.06576](https://arxiv.org/abs/1508.06576)
- only transform 1 pic
- input is 0-1
- gram matrix is different
- loss function source is different

### 4 neural style transfer methods
1. optimizer method
1. vgg16 one style one model
1. wct
1. wct2

### code todo
- make a simple usable code for each method