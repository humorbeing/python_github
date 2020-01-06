# Plan
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

# Problem
- 1-i-abc is too big. The plan was to make 1-ii-abc,
but, i am going to make cnn version first.
    - done
- organize coding files