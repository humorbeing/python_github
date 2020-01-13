# First 5 vr loss log
- check seed: run same code twice. on seed=1.
- S_t is not involved in training S_t+1.
    - with or without torch.no_grad().
- these are all fresh models.

# moved onto RNN
- first, combined with VAE. so trained V+R model
    - It did not work. everything is worse.
- Only included rnn.params in optimizer, given the similar
result with rnn alone model, but with huge pred_loss.

# WHAT I HAVE LEARNED
- one set: turns out the z's are very very different. I didn't understand this stochastic process.
    - I think, try to make a z to z_prediction training is very stupid, that's why when record, they use stochastic, when mdn they use gmm.
- looks like i made a big mistake on how to handling stochastic process.
    - more stochastic correct model
    - non stochastic model
    - rnn need to change the input manners
        - reminds me when design a model, specify inputs like \[batch, squence, data], outputs \[batch, z size]. so when make a multi models, I can coordinate better.
    - recording process needs to be changed
        - maybe i only should save mu and std, not z. and generate on site.
    - when inference, on vae, do i need to use mean value instead of making a stochastic output? should it apply to recording process?
        - what about when training joint models? i need consider these too.
    - looks like i have made series of mistakes
        - inference on vae, shouldn't use training
        - joint model, use stochastic vs deterministic
- need to look into world model codes
    - i am lack of the knowledge of how to make stochasicity work right
    - look into world models dreaming part.
    - check the option for mdn loss being curiosity
- NEED more code reading
- Need cleaning code.
- add folder to yyyymmddttttttt-log-loss
    - then, unify with log.txt as log name
    - also, copy of the executed code should be included

# ideas 
- [x] change log name as date + content: since i am taking a note in log files.
- [x] realized S_t need it's gradient to train VAE, not only S_t+1
    - [ ] need S_t kl and r loss too.
- at this point, i think pred loss is dominating whole training process
    - pred loss is bad idea.
    - [x] let me try to get rid of this.
    - Question. Why pred recon loss is low???????? WTF???
    - **BIG CONCLUSION:** That's why, I was wondering why when make record, use mu and stddiv to make z.
    maybe because, for outputs-> z is stochatic, but pred z is determinstic. what it means, z is
    very, but with these z, decoder can always make sense of it. so pred loss is huge, but pred recon loss is small.
    - [ ] need to check this idea, if this is true, then when making recorde, i need to make some changes to the proceedure
- [x] should i take note here?
- make a alone rnn, that gives exact result as rnn_me (check math)
- seperate optimizer
- rv model, train with one l oss at a time (kl -> kl + recon -> kl + recon + mdn ->...)
- rv model, more weight to RNN model, if competing , then RNN = 10 VAE
- I am too limited to VAE, can't really do anything other then this.
- vae: cnn + rnn to produce mean and std, and rnn + cnn to decode.

# NOTE
- logs/20181119-07-14-30-vr_loss.txt
    - this is joint training.
pred loss is 4300 to 236, and everything else is messed up?
i think pred loss is make too big of a mess.
    - first, make a rnn alone structure in VR, that gives similar result as RNN_me
- logs/20181119-07-37-27-vr_loss.txt
    - reprecate result as in rnn_me
    - mdn is almost same.
    - look like I messed up pred loss.
    - problem is prediction loss is way too high
        - in rnn_me, it's 0.445
        - in vr_rnn along, it's 19095
        - why? looks like a math problem.
- logs/20181119-08-15-43-vr_loss_alone_rnn.txt
    - add rnn._init(), same result
    - pred loss is too big
    - want to make single dataset
- logs/20181119-11-54-47-vr_loss_alone_rnn.txt
    - deleted pred reconstruction loss
    - result is as worse as original version.
- logs/20181119-11-59-43-vr_loss_alone_rnn.txt
    - one set
- logs/20181119-12-00-32-rnn_loss.txt
    - one set
- logs/20181119-12-28-52-vr_loss_no_pred.txt
    - no prediction loss
    - looks like its training, finally
    - maybe i should try weight rnn more
- logs/20181119-12-38-05-vr_loss_no_pred.txt
    - no pred, 200 epoch, AND mdn_loss *= 10
- logs/20181119-12-54-33-vr_loss_no_pred.txt
    - looks like rnn have to compromise itself for kl loss
    - not what i wanted, how about more weight to rnn?
- logs/20181119-13-10-43-vr_loss_no_pred.txt
    - with 100x mdn loss, things are bed, mdn loss is in negative region.
- logs/20181119-13-26-07-vr_loss_no_pred_now_loss.txt
    - now and next loss: looks like a waste of computation