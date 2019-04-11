`FORMAT---------------------------------------`
# NAME of log
- where is it computed
- code: executed code (copy relative path)
- log: log location (copy relative path)
- time: Start
`FORMAT----------------------------------------`
# CL_trainer
1. cl good 1 load freeze
    - pnu my
    - code: cl_trainer
    - setup: cl good1 model, load, freeze
    
# RUNNING encoder-decoder-trainer with rnn, freeze
1. encoder load, freeze on
    - gcp pytorch2
    - code:
    - log:
    - setups: encoder load, freeze on
    - time : 2 pm (all)
2. encoder load, freeze off
    - gcp cpu8
    - setup: encoder load, freeze off
3. no load, no freeze
    -gcp cpu4
    -setup: no load, no freeze


# Competing loss, max limited
- pnu my
- code: EXPERIMENT_RUN/a3c_small_domain/CL_limit_max/CL_limit_max.py
- log: EXPERIMENT_RUN/a3c_small_domain/CL_limit_max/logs

# competing loss, find lambda log
- pnu my
- code: EXPERIMENT_RUN/a3c_small_domain/competing_loss_WM/encoder_model_v1001.py
    - the lambda is varied {0.2, 0.5, 0.7, 1}
- log: EXPERIMENT_RUN/a3c_small_domain/competing_loss_WM/logs
- time: 2.33 pm

# max action
- GCP cpu8-1
- code: EXPERIMENT_RUN/a3c_small_domain/max_action/GCP_cpu8-1_run.py
- log: EXPERIMENT_RUN/a3c_small_domain/max_action/logs/20190206-23-19-07-GCP_C81_max_action.txt
- time: 8:21 am
- note: stopped accidentally
- Pen9
- code: EXPERIMENT_RUN/a3c_small_domain/max_action/Pen9_run.py
- log: 20190207-11-38-31-Pen8_max_action.txt
- time: 11:43 am

# Random Seed 100
- PNU my
- code: EXPERIMENT_RUN/a3c_small_domain/different_seed/PNU_my_runner.py
- log: EXPERIMENT_RUN/a3c_small_domain/different_seed/logs/20190207-07-47-51-PNUmy_seed_100.txt
- time: 7:56 am

# Random Seed 1000
- GCP pytorch-2
- code: EXPERIMENT_RUN/a3c_small_domain/different_seed/GCP_pyt_run.py
- log: EXPERIMENT_RUN/a3c_small_domain/different_seed/logs/20190206-23-20-49-GCP_pytorch-2_1000.txt
- time: 8:21 am

# 3 actions Small, Simple
- n202
- code: EXPERIMENT_RUN/a3c_small_domain/3_small_simple/this_models.py
- log: EXPERIMENT_RUN/a3c_small_domain/3_small_simple/logs/20190205-16-40-10-3_SS.txt

# Small Simple Domain
- PNU my
- code: EXPERIMENT_RUN/a3c_small_domain/small_simple/PNU_my_small_simple.py
- log: EXPERIMENT_RUN/a3c_small_domain/small_simple/logs/20190205-12-43-50-PUNmy_simple_domain.txt

# Extreme RNN
- GCP cpu8-1
- code: EXPERIMENT_RUN/a3c_small_domain/extreme_rnn/GCP_cpu8_run.py
- log: EXPERIMENT_RUN/a3c_small_domain/extreme_rnn/logs/20190205-02-42-33-extreme_rnn.txt

# untrained encoder
- PNU my
- code: EXPERIMENT_RUN/a3c_small_domain/Encoder/PNU_untrained_encoder_a3c.py
- log: EXPERIMENT_RUN/a3c_small_domain/Encoder/logs/20190205-08-40-47-untrainedencoder_PNUmy.txt

# VAE RNN a3c small domain
- n202
- code: /VAE/vae_rnn_a3c.py
- log: /mnt/D8442D91442D7382/Mystuff/Workspace/python_world/python_github/EXPERIMENT_RUN/a3c_small_domain/VAE/logs/20190204-19-48-23-a3c_baselines_small_domain.txt

# ENCODER RNN a3c small domain
- GCP CPU8-1
- code: EXPERIMENT_RUN/a3c_small_domain/Encoder/GCP_cpu8_run.py
- log: EXPERIMENT_RUN/a3c_small_domain/Encoder/logs/20190204-11-59-46-encoder_rnn.txt

# more rnn a3c small domain
- GCP pytorch-2
- code: EXPERIMENT_RUN/a3c_small_domain/more_rnn/GCP_RUN.py
- log: EXPERIMENT_RUN/a3c_small_domain/more_rnn/logs/20190204-07-44-53-a3c_baselines_small_domain.txt

# RNN NO limit a3c small domain
- PNU my PC
- code: EXPERIMENT_RUN/a3c_small_domain/RNN/pnu_Nolimit_a3c_small_domain_rnn_only.py
- log: EXPERIMENT_RUN/a3c_small_domain/RNN/logs/20190204-17-09-46-a3c_baselines_small_domain.txt

# RNN but limited to 10000 a3c small domain
- PNU my PC
- code: EXPERIMENT_RUN/a3c_small_domain/RNN/run_pnu_a3c_small_domain_rnn_only.py
- log: EXPERIMENT_RUN/a3c_small_domain/RNN/logs/20190204-04-45-50-a3c_baselines_small_domain.txt

# Pixel - Pixel a3c small domain
- n202
- code: EXPERIMENT_RUN/a3c_small_domain/Pixel/pixel_a3c_v01.py
- log: EXPERIMENT_RUN/a3c_small_domain/Pixel/logs/20190204-05-40-18-a3c_baselines_small_domain.txt

# Pixel with RNN a3c small domain
- GCP pytorch-2
- code: EXPERIMENT_RUN/a3c_small_domain/Pix_RNN/GCP_pix_rnn_a3c.py
- log: EXPERIMENT_RUN/a3c_small_domain/Pix_RNN/logs/20190203-23-59-18-a3c_baselines_small_domain.txt

# MLP a3c small domain
- n202
- code: EXPERIMENT_RUN/a3c_small_domain/MLP/a3c_MLP_expriment.py
- log: EXPERIMENT_RUN/a3c_small_domain/MLP/20190203-18-56-30-a3c_baselines_small_domain.txt

