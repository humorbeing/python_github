from math import sqrt
import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            #euclidean_distance = sqrt( (features[0]-predict[0])**2 + (features[1]-predict[1])**2 )
            #euclidean_distance = np.sqrt( np.sum((np.array(features)-np.array(predic))**2) )
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]



    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    #print(vote_result, confidence)
    return vote_result, confidence

df = pd.read_csv("data.txt")
df.replace('?',-99999, inplace=True)
df.drop(['id'],1,inplace=True)
full_data = df.astype(float).values.tolist()
#full_data = df.values.tolist() #if no float type, some numbers will be string like '10',10,7

#print(full_data[:10]) #first 10. so maybe [10:]last 10?
#print(full_data[:5])
#full_data = random.shuffle(full_data)#not in this case.
accuracies = []
for j in range(25):

    random.shuffle(full_data)
    #print(20*'#')
    #print('='*20) #interesting print
    #print(full_data[:5])

    test_size = 0.2 #if more test? less 1.0 confi error?
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote,confidence = k_nearest_neighbors(train_set, data, k=5) #5 is good
            #if k is too big, will do more harm then good.
            if group == vote:
                correct += 1
            else:
                pass
                #print(confidence)
            total +=1

    #print('Accuracy:', correct/total)
    accuracies.append(correct/total)
print(sum(accuracies)/len(accuracies))
