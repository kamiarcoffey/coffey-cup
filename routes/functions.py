import json
import requests
import lxml.html as lh
import pandas as pd
import numpy as np
import webbrowser

from flask import Flask
from flask import Flask, render_template

app = Flask(__name__)

FPL_URL = 'https://fantasy.premierleague.com/drf/bootstrap-static'

HEADER = """<!-- DOCTYPE HTML -->
<style>
table {
  align-text: center;
  border: 1px solid black;
  border-collapse: collapse;
  margin: 0px;
  padding: 0px;
}

td, th {
  white-space: nowrap;
  border: 1px solid black;
}

</style>
<html>
    <head>The Draft Room!
    </head>
    <body>
        <p>Welcome!</p>
        <div>
            <table style="width:30%">
                <tr>
                <th>Name</th>
                <th>Points Per Minute</th>
                </tr>
"""
FOOTER = """
            </table>
        </div>
    </body>
</html>
"""

def readAPItoFile():
    r = requests.get(FPL_URL)
    jsonResponse = r.json()
    with open("data/leagueInfo.json", 'w') as outfile:
    	json.dump(jsonResponse, outfile)

def getAllPlayersDetailedJson():
	with open('data/leagueInfo.json') as json_data:
		dataObject = json.load(json_data)
		return dataObject

def writeToHTMLTable(fileName, list):
    f = open(fileName,'w')
    f.write(HEADER)
    for element in list:
        name = element[0]
        ppm = str(element[1])
        f.write('<tr><td>')
        f.write(name)
        f.write('</td><td>')
        f.write(ppm)
        f.write('</td></tr>')
    f.write(FOOTER)
    f.close()

def flaskReturn(list):
    string = HEADER
    for element in list:
        name = element[0]
        ppm = str(element[1])
        string = string + '<tr><td>'
        string = string + name
        string = string + '</td><td>'
        string = string + ppm
        string = string + '</td></tr>'
    return string

def getListofPlayers():
	allPlayers = getAllPlayersDetailedJson()
	namelist = []
	for entry in allPlayers["elements"]:
		namelist.append([entry['first_name'],entry['second_name']])
	# htmlcode = html.table(namelist, header_row=['First name',   'Last name'])
	return nameList

def findBestRatioPlayer():
    allPlayers = getAllPlayersDetailedJson()
    points_per_min = []
    for entry in allPlayers["elements"]:
        # format: ['first_name last_name', points_per_min]
        if entry['minutes'] > 0:
            points_per_min.append([entry['first_name']+' '+entry['second_name'], (entry['total_points']/entry['minutes'])])
    points_per_min.sort(key=lambda x: x[1], reverse=True)
    return points_per_min
    # for element in points_per_min:
    #     print(element[0],element[1])
