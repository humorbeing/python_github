# Experiment RUN
- log name: Pong-D-4-nogamma-nopopvariance
- env: Pong-ramDeterministic-v4
- no gamma standardization (only obs, no rewards)
- standardization is population variance

# Plan
- [ ] input data processing
    - [ ] re implement vec_normali.
        - [x] without gamma for return, i guess
            - so, there is an complication about calculating returns in vecnorm
            implementation from openai's baselines.
            - meanwhile, if i don't do gamma. then it seems only obs are being normalized.
            - [x] evaluation without gamma
            - [ ] in running mean std: experiment on sample variance vs population variance
        
- [ ] input data processing in evaluation
- [ ] v0 vs deterministic v4 experiment

# Done in Last version (v3)
- [x] logging system, from a3c
- [x] evaluation, learn from pytoch ppo code
    - I was logging training errors from last 10 episodes
    - in the code, there is a evaluation code, learning it.    


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