#!/usr/bin/env python

# a list of all include statements for the entire notebook - re-run this cell if code is edited further down
import json
import requests
import lxml.html as lh
import operator
import pandas as pd
import numpy as np
from scipy import special
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt
from collections import defaultdict


# FPL data is dumpted here, but it is aggreated not by player
STATS_URL = 'https://fantasy.premierleague.com/drf/bootstrap-static'

# FPL data is dumpted here by player where playerNumber is appended to the end of the API call string
PLAYER_URL = "https://fantasy.premierleague.com/drf/element-summary/" # [playerNumber]


# better not to querry the API a million times - read it to a file
# def readAPItoFile():
#     r = requests.get(STATS_URL)
#     jsonResponse = r.json()
#     with open("leagueInfo.json", 'w') as outfile:
#         json.dump(jsonResponse, outfile)

# readAPItoFile()


# turn the JSON into an object
def getAllPlayersDetailedJson():
    with open('leagueInfo.json') as json_data:
        dataObject = json.load(json_data)
        return dataObject

dataObject = getAllPlayersDetailedJson()


# just for fun, lets get a list of player names. I think I'll need their playerID as the primay key, but gotta make sure this works
def getListofPlayers():
    allPlayers = getAllPlayersDetailedJson()
    namelist = []
    for entry in allPlayers["elements"]:
        namelist.append([entry['first_name'], entry['second_name']])
    return nameList

def playersToPoints():
    allPlayers = getAllPlayersDetailedJson()
    pointsPerPlayer = {}
    for entry in allPlayers["elements"]:
        pointsPerPlayer[entry['web_name']] = entry['total_points']
    return pointsPerPlayer

def playersToId():
    allPlayers = getAllPlayersDetailedJson()
    dict = {}
    for entry in allPlayers["elements"]:
        dict[entry['second_name']] = entry['id']
    return dict

def idToPlayer():
    allPlayers = getAllPlayersDetailedJson()
    dict = {}
    for entry in allPlayers["elements"]:
        dict[entry['id']] = entry['second_name']
    return dict

def findBestRatioPlayer():
    allPlayers = getAllPlayersDetailedJson()
    points_per_min = []
    for entry in allPlayers["elements"]:
        # format: ['first_name last_name', points_per_min]
        if entry['minutes'] > 0:
            points_per_min.append([entry['id'], entry['first_name']+' '+entry['second_name'], (entry['total_points']/entry['minutes'])])
    points_per_min.sort(key=lambda x: x[1], reverse=True)
    return points_per_min

def findTotalPoints():
    allPlayers = getAllPlayersDetailedJson()
    points_per_min = []
    for entry in allPlayers["elements"]:
        # format: ['first_name last_name', points_per_min]
        points_per_min.append([entry['web_name'], entry['total_points']])
    points_per_min.sort(key=lambda x: x[1], reverse=True)
    return points_per_min

ppm_df = pd.DataFrame(findBestRatioPlayer(), columns=['ID', 'Player Name', 'FPL points/minute'])
ppm_df.head(15)

totalPoints_df=pd.DataFrame(findTotalPoints(), columns=['Player Name', 'FPL Points So Far'])


featuresVector = ['goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 'own_goals','penalties_saved','penalties_missed',
        'yellow_cards','red_cards', 'saves', 'open_play_crosses','big_chances_created','clearances_blocks_interceptions',
        'recoveries','key_passes', 'tackles','winning_goals','attempted_passes','completed_passes', 'penalties_conceded',
        'big_chances_missed','errors_leading_to_goal','errors_leading_to_goal_attempt','tackled','offside','target_missed',
        'fouls','dribbles','open_play_crosses','big_chances_created',' clearances_blocks_interceptions','recoveries',
        'key_passes','errors_leading_to_goal','errors_leading_to_goal_attempt']


'''THIS API CALL WILL TAKE 5 MINUTES. DON'T RE-RUN UNLESS NECESSARY!
the data will be saved to the playerVectorDictionary.json file,
and can be reloaded via the getVectorsFromJSON functon'''

# playerDataDictionary = dataObject['elements']
# playerNames = {}

# for i in playerDataDictionary:
#     playerNames[i['id']] = i['web_name']

# playerVectorDictionary = defaultdict(dict)

# for player in playerNames:
#     r = requests.get(PLAYER_URL + str(player))
#     data = json.loads(r.text)['history'] # this is a list of dictionary entries
#     for entry in data:
#         entryVector = [entry[feature] for feature in featuresVector]
#         playerVectorDictionary[playerNames[player]] = list(map(operator.add, playerVectorDictionary.setdefault(playerNames[player], [0]*len(entryVector)), entryVector)) # accumulate each game entry

# playerVectorDictionary_json = json.dumps(playerVectorDictionary)
# f = open("playerVectorDictionary.json","w")
# f.write(playerVectorDictionary_json)
# f.close()


def getVectorsFromJSON():
    with open('playerVectorDictionary.json') as json_data:
        dataObject = json.load(json_data)
        return dataObject

playerVectorDictionary = getVectorsFromJSON()


x = np.array(list(playerVectorDictionary.values()))
y = np.array([[playersToPoints()[player]] for player in playerVectorDictionary.keys()])
print("check that vector sizes match!!", len(y), "=?", len(x))


# Some helper math functions

def CrossEntropy(yHat, y):
    if y == 1:
        return -log(yHat)
    else:
        return -log(1 - yHat)

def sigmoidDerivative(x):
    return x * (1.0 - x)


class NeuralNet:
    def __init__(self, x, y, iterations = 100, learningRate=.33): # give it some default parameters
        self.n_in = len(x[0]) # make the dimensionality dependent on the inputs
        self.n_out = 1
        self.n_hidden = len(x)
        size1 = (self.n_in, self.n_hidden)
        size2 = (self.n_hidden, self.n_out)
        self.weights1 = np.random.uniform(-.01, .1, size1) # weights from input to nodes
        self.weights2 = np.random.uniform(-.01, .1, size2) # weights from nodes to output
        self.iterations = iterations
        self.learningRate = learningRate

    def sigmoidDerivative(self, x): # basic calculus
        return x * (1 - x)

    def feedForward(self, X): # an implementation of the above equation
        self.z = scipy.special.expit(np.dot(X, self.weights1)) # X is the vector being fed forward
        self.z_prime = np.dot(self.z, self.weights2) # has to be a parameter because we have ver 500 players to train with
        return scipy.special.expit(self.z_prime)

    def backProp(self, x, y, update):
        self.loss = y - update # loss needs to be stored as a data member because back prop needs to re-access it
        self.loss_adjustment = self.loss * self.sigmoidDerivative(update)
        self.z_error = self.loss_adjustment.dot(self.weights2.T)  # implementation of text book code
        self.z_adjustment = self.z_error * self.sigmoidDerivative(self.z) * self.learningRate
        self.weights1 += x.T.dot(self.z_adjustment)
        self.weights2 += self.z.T.dot(self.loss_adjustment)

    def train(self, x, y):
        for _ in range (self.iterations):
            update = self.feedForward(x)
            self.backProp(x, y, update)

    def run(self, testVector):
        sum =0
        for i in range (len(self.weights1)):
            sum += testVector[i] * self.weights1[i][0]
        return sum


expectedValue_LR = []
# run a loop where each iteration is training with a different learning rate
for i in range (1, 20):
    learningRate = (1/i)
    NN = NeuralNet(x, y, 500, learningRate)
    NN.train(x, y)
    inputVector = np.array(([19, 7, 7, 37, 0, 0, 1, 0, 0, 0, 8, 6, 25, 60, 30, 11, 4, 619, 463, 0, 16, 0, 0, 52, 23, 27, 13, 22, 8, 6, 25, 60, 30, 0, 0]))
    predictedPoints = NN.run(inputVector)
    expectedValue_LR.append((learningRate, predictedPoints))


plt.plot(*zip(*expectedValue_LR))
plt.ylabel('Converged Expected FLP Points')
plt.xlabel('Learning Rate Used In Model')
plt.title('Examination of Learning Rate Impact On Model Output')
plt.show()


expectedValue_Trials = []
for i in range (1, 1000, 100):
    NN = NeuralNet(x, y, i, .3)
    NN.train(x, y)
    inputVector = np.array(([19, 7, 7, 37, 0, 0, 1, 0, 0, 0, 8, 6, 25, 60, 30, 11, 4, 619, 463, 0, 16, 0, 0, 52, 23, 27, 13, 22, 8, 6, 25, 60, 30, 0, 0]))
    predictedPoints = NN.run(inputVector)
    expectedValue_Trials.append((i, predictedPoints))


plt.plot(*zip(*expectedValue_Trials))
plt.ylabel('Converged Expected FLP Points')
plt.xlabel('Number Of Trials Used In Model')
plt.title("Modeling Network Trial Counts")
plt.show()


learningRate = 0.3
n_trials = 100
NN = NeuralNet(x, y, n_trials, learningRate)
NN.train(x, y)
inputVector = np.array(([19, 7, 7, 37, 0, 0, 1, 0, 0, 0, 8, 6, 25, 60, 30, 11, 4, 619, 463, 0, 16, 0, 0, 52, 23, 27, 13, 22, 8, 6, 25, 60, 30, 0, 0]))
predictedPoints = NN.run(inputVector)
print("The exptected FPL points earned is: ",predictedPoints)


totalPoints_df.head(20)


learningRate = 0.45
n_trials = 1000
NN = NeuralNet(x, y, n_trials, learningRate)
NN.train(x, y)

playerModeledPoints = []
for player in playerVectorDictionary:
    testArray = playerVectorDictionary[player]
    predictedPoints = NN.run(testArray)
    playerModeledPoints.append((player, predictedPoints))


playerModeledPoints.sort(key=lambda x: x[1], reverse=True)
playerModeledPoints_df = pd.DataFrame(playerModeledPoints, columns = ['Player Name', 'Predicted Points'])
playerModeledPoints_df.head(20)


actual = dict(findTotalPoints())
predicted = dict(playerModeledPoints)
comparison = {}
for player in predicted.keys():
    comparison[player] = (predicted[player], actual[player])

predicted, actual = zip(*comparison.values())
t2, p2 = stats.ttest_ind(predicted, actual)
print("t test:",t2, "p value:",p2)


plt.figure(figsize=(30,15))
plt.plot(predicted, actual, 'ro')
plt.xlabel("Predicted FPL Points", fontsize=20)
plt.ylabel("Actual FPL Points", fontsize=20)
plt.title("FPL Points By Player", fontsize=25)
plt.show()
