# Plan
- [ ] logging system
    - [x] simple one. from my A3C experiment
    - [ ] pytorch-ppo code has its own logging system
    attached to the evaluation process. learning it.
- [x] evaluation, learn from pytoch ppo code
    - I was logging training errors from last 10 episodes
    - in the code, there is a evaluation code, learning it.    
- [ ] GPU
- [ ] need to fix python FILE structure
    - run_this.py is outside, rest of them in a folder

# v2 is deserted (given up) version.
- given up on pytorch free env
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