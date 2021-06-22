# Kamiar Coffey
# Coffey-Cup README

👀 Check out the [Code WalkThrough And Discussion](https://github.com/kamiarcoffey/coffey-cup/blob/master/Code%20WalkThrough%20And%20Discussion.ipynb) for a tour 

An excuse to learn how to use Flask: currently a preliminary set-up with Flask to use the Premier League API to create a web-app that will run predictive metrics to help pick players for the Coffey Cup fantasy draft league

## Code ##
* All code is in AI_Engine.py. To run via the terminal use Python3
* It includes some API calls and myPlotLib packages
* It also references some JSON files located in the main directory

## Data Files ##
* leagueInfo.json stores info on the league as a whole (ref player name to ID)
* playerVectorDictionary.json stores the working JSON object for the training vectors of each player used in the model
* values in this dictionary are summed across the entire season already

# leagueChampions

## To Run ##
```
bash
$ export FLASK_APP=coffey-cup.py
$ flask run
```

