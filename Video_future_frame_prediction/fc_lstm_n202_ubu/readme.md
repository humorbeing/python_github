### Plan
1. Input is vector. Means no CNN
    1. neither decoder nor predictor have no generated input
        1. encoder - decoder         
        1. encoder - predictor            
        1. encoder - (decoder and predictor)
    1. decoder and predict have generated input
        1. encoder - decoder         
        1. encoder - predictor            
        1. encoder - (decoder and predictor)        
1. Input is matrix. Means need CNN
    1. neither decoder nor predictor have no generated input
        1. encoder - decoder         
        1. encoder - predictor            
        1. encoder - (decoder and predictor)
    1. decoder and predict have generated input
        1. encoder - decoder         
        1. encoder - predictor            
        1. encoder - (decoder and predictor)        

### Problem
- 1-i-abc is too big. The plan was to make 1-ii-abc,
but, i am going to make cnn version first.
    - done
- organize coding files

### combination
decoder is zero input
- models
    - lstm encoder decoder
    - lstmcell encoder decoder
    - cnn lstmcell encoder decoder v0001
    - cnn lstmcell encoder decoder v0002
    - cnn-lstmcell-v0003
    - cnn flatten lstmcell encoder decoder
- experiments
    - recon
    - order recon
    - pred
- combo
    - [x] lstm recon
    - ~~lstm order-recon~~
    - ~~lstm pred~~
    - [x] lstmcell recon
    - [x] lstmcell order-recon
    - [x] lstmcell pred
    - [x] cnn-lstmcell-v0001 recon x 2
    - cnn-lstmcell-v0001 order-recon
    - cnn-lstmcell-v0001 pred
    - [x] cnn-lstmcell-v0002 recon
    - cnn-lstmcell-v0002 order-recon
    - [x] cnn-lstmcell-v0002 pred
    - [x] cnn-flatten-lstmcell recon
    - cnn-flatten-lstmcell order-recon
    - [x] cnn-flatten-lstmcell pred
    - [x] cnn-lstmcell-v0003 recon
    - [x] cnn-lstmcell-v0003 order-recon
    - [x] cnn-lstmcell-v0003 pred
    - cnn-flatten recon (-0.5,0.5) with tanh
    - cnn-v0003 recon (55)
    - lstmcell recon (55)
    - lstmcell-v0002 recon
    - lstmcell-v0002 recon (55)
    - lstmcell-v0002 order-recon
    - [x] [colab?] lstmcell-v0002 pred
    - [x] [colab?] lstmcell recon
    - lstmcell-v0001 EDP

### after read the original paper code
- lstm is 4096 to 2048
    - en1 4096 to 2048
    - en2 2048 2048
    - de1 4096 2048
    - de2 2048 4096
    - pre is same as de
- loss function is cross entropy Bernoulli
- input is 0-1
- experiments
    - [x] recon
    - [x] pred

### it's not learning at all
with all these hustle, right number of inputs and outputs
for LSTMs, use binary cross entropy, input is set to (0,1),
so is torch.sigmoid attached to output layer. the out come is
it's not learning at all.
- I think it's a good opportunity explore optimizers and learning rate
schedulers. 
- Let's do "out of the box performance"
    - [pnu2] all optimizer
- how about some loss change on original model?
    - options are: (-1,1) (0,1), tanh, sigmoid, bce, mse
        - (-1,1), tanh, mse
        - (-1,1), non, mse
        - (0,1), sigmoid, bce
        - (0,1), sigmoid, mse
        - (0,1), non, mse            
    - [pnumy] 01, mse loss, with sigmoid, recon
    - (-1,1), tanh, mse
    - (-1,1), non, mse
    - (0,1), sigmoid, bce
    - (0,1), non, mse  
         
- recon as input of lstm
    - [colab] bce loss, with sigmoid, recon, recon_in

### Make a args lead experiment system
- done
    - made a 4096 4096 lstmcell
    - combos
        - zero in, output in
        - recon, pred
        - last active
            - (-1,1), tanh, mse
            - (-1,1), non, mse
            - (0,1), sigmoid, bce
            - (0,1), sigmoid, mse
            - (0,1), non, mse
- experiments schedule
    - [x] zero in, recon, (-1,1), tanh, mse
    - output in, recon, (-1,1), tanh, mse
    - [x] zero in, pred, (-1,1), tanh, mse
    - output in, pred, (-1,1), tanh, mse

### All fail to train the model
- looks like without last layer activation function is way to go
- and also add both losses tracking, and use one of them to train
- make a all working EDP network and runner
- experiments schedule
    - [x] [colab Geem] on lstm copy model, recon, loss combo
        - (-1,1), tanh, mse
        - (-1,1), non, mse
        - (0,1), sigmoid, bce
        - (0,1), sigmoid, mse
        - (0,1), non, mse
    - [colab apollo] on lstm copy model, both, loss combo
        - [x] (-1,1), tanh, mse
        - [x] (-1,1), non, mse
        - [x] (0,1), sigmoid, bce
        - [ ] (0,1), sigmoid, mse
        - [x] [colab Geem] (0,1), non, mse
    - [x] [pnumy] on lstm copy model, pred, loss combo
        - (-1,1), tanh, mse
        - (-1,1), non, mse
        - (0,1), sigmoid, bce
        - (0,1), sigmoid, mse
        - (0,1), non, mse
    - [x] [pnu2] on lstm copy model, recon, loss combo, zero_input false
        - (-1,1), tanh, mse
        - (-1,1), non, mse
        - (0,1), sigmoid, bce
        - (0,1), sigmoid, mse
        - (0,1), non, mse
    - [x] [colab humor] on lstm copy model, pred, loss combo, zero_input false
        - [x] (-1,1), tanh, mse
        - [x] (-1,1), non, mse
        - [x] (0,1), sigmoid, bce
        - [x] (0,1), sigmoid, mse
        - [x] (0,1), non, mse
    - [colab ray] on lstm copy model, both, loss combo, zero_input false
        - [x] (-1,1), tanh, mse
        - [x] (-1,1), non, mse
        - [x] (0,1), sigmoid, bce
        - [ ] (0,1), sigmoid, mse
        - [x] [colab humor] (0,1), non, mse


### CNN with flatten, without flatten

### mixed precision training
- use api
- use torch.float16
    - problems: in float16, you can perform certain math operations
        - so, you can't just set default tensor type or dtype to float16
    and everything runs in float16
    - there is GPU support angle, looks like some gpu can't run in float16
    - this way need more experiment
        - like float16 + float16, to what extend
    - go back to API, looks like Nvidia has one


### logging system need better naming
- [x] naming is too similar, or exactly same.
    - I can only differentiate by the SN.

### plots
- [x] 5losses in 6 ways
    - recon, with zero input
    - recon, with new output as input
    - pred, with zero input
    - pred, with new output as input
    - both, with zero input
    - both, with new output as input
- [x] all recons and all pred
    - which metric? hmm, 0 n m
        - recon, with zero input
        - recon, with new output as input
        - recon from both, with zero input
        - recon from both, with new output as input
    - preds
        - pred, with zero input
        - pred, with new output as input
        - pred from both, with zero input
        - pred from both, with new output as input

- [x] sigmoid mse train vs bce train
    - mse metric
        - all recons
        - all preds
    - bce metric
        - all recons
        - all preds

### copy paper setups
- Try to train model with sigmoid and bce loss.
    - pre setups: 
        - no standardization
        - no mse loss, only bce
        - optimizer: rmsprop is trigger
        - what model?: pred
        - zero_input: false
        - change naming
- implement
    - [x] optimizer: rmsprop
    - [x] learning rate:
    - [x] regu L2 and decay
    - [x] learning rate decay
    - [x] gradient clip
    - [x] init
- experiment
    - [x] [pnumy] 1000 epoch rmsprop

### Why won't train? IDEA: tanh sigmoid is bad idea
- tanh output is (-1,1), then sigmoid. it will constraint sigmoid
output to (0.269,0.731)
- to implement
    - [x] model with 2 linear layers each side
    - [x] adam with rl
    - [x] all fancy stuff
    - [x] fix sigmoid bce
- experiment
    - [x] [pnu2] rmsprop, both, zf
    - [x] [pnu2] adam, both, zf
    
### investigate output of last layer of LSTM
- is it only between -1, 1.
- to fix it, how about multiply with 10?
- alright, as it turns out, max of and min of lstm is (-0.1, 0.1)
- bright out old guns, lets give a 100x of last layer, then sigmoid

### to be able to run old code
- program with code version.
- code_v0001 folder, so that i can visit anytime.
- make a check list for next project.

### demo maker
- clear all models so far
- use dict to save
- use check point style
- make cross device code
- save as torch.save(model, path) is practice
    - loading part is too tricky


### v0007
- [x] model with multiply args
- [x] args with top 3 losses
    - 0 s m
    - 0 s b
    - 0 n m
- all fancy optimization into args
    - [x] optimizer
    - [x] learning rate:
    - [x] regu L2 and decay
    - [x] learning rate decay
    - [x] gradient clip
    - [x] init
- [x] model save protocol
- [x] trained model demo and save
- [x] naming
- [ ] merge demo and uti
- [ ] fix ax[0].axis('off')
- experiment
    - default settings
        - [x] [colab A] 100s m
        - [x] ~~[pnu2] 100s b~~
            - bugged to 100 M
        - [ ] [pnu2] 100s b
        - ~~s m~~
        - ~~s b~~
        - [x] [pnumy] n m
    - 0 N M looks good
        - [colab H]no clip
        - [colab A] rmsprop
        - [pnumy] no init
### v0008
- bug
    - [x] nameing: bce loss -> B
    - [x] fix train with bce loss
- all model is `Non MSE`
- all model is `both`
- all model is `zero input = False`
- new models
    - [x] lstm_v0001
    - cnn
    - cnn flatten
- experiment
    - [x] [colab R] lstm_v0001 hidden 256
    - [x] [colab R] lstm_v0001 hidden 512
- is epoch too many? this model is almost saturated at epoch 50.