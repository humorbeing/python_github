for param in model.parameters():
		param.requires_grad = False

for param in model.encoder.parameters():
    param.requires_grad = False

'''
model A has pretrained weights.
model A, B share part S.
load model A's pretrained weight,
copy A's S's weight to model B's S.
then freez model B's S.
'''

A = modelA()

for param in A.S.parameters():
    param.requires_grad = False

A.load_state_dict(torch.load(ENCODER_MODEL_PATH))

for param in A.S.parameters():
    param.requires_grad = False

B = modelB()

for param in B.S.parameters():
    param.requires_grad = False

B.S.load_state_dict(A.S.state_dict())
for param in B.S.parameters():
    param.requires_grad = False

for param in B.S.parameters():
    print(param)

'''
result:
forb A, doesn't work
forb A load, doesn't work
forb B, work
forb B, load, work.
'''