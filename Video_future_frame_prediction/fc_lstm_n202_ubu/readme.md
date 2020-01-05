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
    - [pnu2] cnn-lstmcell-v0002 pred
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
    - [colab?] lstmcell-v0002 pred
    - [colab?] lstmcell recon
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
    - [pnumy] recon
    - pred

### it's not learning at all
with all these hustle, right number of inputs and outputs
for LSTMs, use binary cross entropy, input is set to (0,1),
so is torch.sigmoid attached to output layer. the out come is
it's not learning at all.
- I think it's a good opportunity explore optimizers and learning rate
schedulers. 
- Let's do "out of the box performance"