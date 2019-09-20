# Dont run this v2 version
- given up on attempt 
# Giving up on pytorch free env
- while trying to do this, if env is numpy
then, when collecting actions without grad part
will involve converting torch V value, action and log action
into numpy. also, when back-prop from all numpy to torch
need to convert back. so pytorch env makes more sense.

# Doing What Now
- [x] pytorch separation from envs
- [ ] rollout pytorch free
    - [x] init pytorch free
    - [ ] insert free? maybe
- [ ] actor_critic numpy to pytorch
    - [ ] in MLP. forward method, numpy to pytorch
        - no fancy, only change
    - [ ] no grad for acting
 
# Trying to add some functions
- pytorch separation?
- a3c in this framework?

# This is most simple version of pytorch PPO
- No RNN
- No GAE
- No V function fancy error
- No Standardization
    - No input
    - No advantage value
- No Pixel
- No 4 frames
- No fancy logging system
- No GPU
- No fixing pytorch env data stream