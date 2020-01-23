### pytorch examples: world language model with wandb
- pytorch
- LSTM, GRU, RNN with relu, transformers
- interesting training process
    - learning rate associate with best validation result
    - keyboard interruption
- testing
- wandb


### experiment
- try various models, and compare the data efficiency
- experiments
    - lstm
    - gru
    - rnn
    - rnn relu
    - transformer(given)
    - transformer from harvard
    - bert?
    - transformer xl
    - reformer

### todo
- [x] lr is adjusted by valid loss and lr scheduler
- [x] early stop
- wandb
- [x] [tested] [added] manual update vs adams update
    - why learning rate is so different?
        - lr 20 is overflow for Adam
        - somehow lr 20 is working with manual update
        - lr 0.001 is used in Adam
    - tested but not working properly, very strange behavior
        - Find: 
            - scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
                - works as intended
            - scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
                - strange behavior