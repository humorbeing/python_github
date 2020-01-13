# if model is trained on GPU
model.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))
# if model is trained on CPU
model.load_state_dict(torch.load(MODEL_PATH))

torch.save(model.state_dict(), MODEL_PATH+save_name)