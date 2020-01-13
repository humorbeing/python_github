import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean,median
from collections import Counter

LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000
in_games = 200
'''
def some_random_games_first():
    for episode in range(50):
        print('game:',episode)
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

some_random_games_first()
'''

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation,reward,done,info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]  #what if doing nothing?

                training_data.append([data[0],output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    #print('Average accepted score:', mean(accepted_scores))
    #print('Median accepted score:', median(accepted_scores))
    #print(Counter(accepted_scores))

    return training_data

def neural_network_model(input_size):
    network = input_data(shape = [None, input_size, 1], name = 'input')

    network = fully_connected(network, 300, activation='relu')
    network = dropout(network, 0.9)

    network = fully_connected(network, 200, activation='relu')
    network = dropout(network, 0.9)

    network = fully_connected(network, 1000, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 200, activation='relu')
    network = dropout(network, 0.5)

    network = fully_connected(network, 500, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    Y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))

    model.fit({'input':X}, {'targets':Y}, n_epoch=4, snapshot_step=500,
              show_metric=True, run_id='openaistuff')

    return model



#model.save('sss.model')
#model.load('sss.model') #model code and input size
def lets_do_it(model,cc):
    scores = []
    #choices = []
    accepted_scores =[]
    for each_game in range(in_games):
        score = 0
        game_memory=[]
        prev_obs = []
        env.reset()
        #print('do game:',each_game)
        #for _ in range(goal_steps):
        for _ in range(int(cc)+200):
            #if isshow:
            #    env.render()
            if len(prev_obs) == 0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs),1))[0])
            #choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                break
        if score >= cc:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]  #what if doing nothing?

                training_data.append([data[0],output])

        scores.append(score)
    if len(accepted_scores) > 0:
        print('Highest:', np.sort(accepted_scores)[len(accepted_scores)-1])
    print('Average Score', sum(scores)/len(scores))
    #print('Choice 1: {}, Choice 0: {}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
    av = sum(scores)/len(scores)
    print('{} out of 200 made -{}- line'.format(len(accepted_scores),cc))
    return training_data,av

def show_me(model):
    scores = []
    #choices = []
    #accepted_scores =[]
    for each_game in range(100):
        score = 0
        #game_memory=[]
        prev_obs = []
        env.reset()
        #print('do game:',each_game)
        #for _ in range(goal_steps):
        while True:
            #if isshow:
            env.render()
            if len(prev_obs) == 0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs),1))[0])
            #choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            #game_memory.append([new_observation, action])
            score += reward
            if done:
                print('score is',score)
                break

    #print('Highest:', np.sort(scores)[len(scores)-1])
    #print('Average Score', sum(scores)/len(scores))
    #print('Choice 1: {}, Choice 0: {}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
    #av = sum(scores)/len(scores)
    #return training_data,av



training_data = initial_population()
model = train_model(training_data)
av = score_requirement
gtg = True#good to go
i = 0
while gtg:

    training_data, av = lets_do_it(model,av)
    #print("{} trainees and with average of {} to wash out".format(len(training_data), av))
    model = train_model(training_data, model)
    i += 1
    if av > 500 or i >0:
        gtg = False

show_me(model)
#initial_population()

#gether data +=, to 100000, better score, go train, use model  ---repeat
