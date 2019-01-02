import gym
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

game_image = gym.game() #pseudo code
np.save('image.save', game_image) # save

game_image = np.load('image.save') # load
dataset = Dataset(game_image) # dataset setup
dataloader = DataLoader(dataset) # dataloader