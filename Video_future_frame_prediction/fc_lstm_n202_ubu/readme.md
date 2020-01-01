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
    - [ ] lstmcell order-recon
    - [x] lstmcell pred
    - [x] cnn-lstmcell-v0001 recon x 2
    - cnn-lstmcell-v0001 order-recon
    - cnn-lstmcell-v0001 pred
    - [x] cnn-lstmcell-v0002 recon
    - cnn-lstmcell-v0002 order-recon
    - cnn-lstmcell-v0002 pred
    - cnn-flatten-lstmcell recon
    - cnn-flatten-lstmcell order-recon
    - cnn-flatten-lstmcell pred
    - cnn-lstmcell-v0003 recon
    - cnn-lstmcell-v0003 order-recon
    - cnn-lstmcell-v0003 pred
    
### Schedule
- work with 
