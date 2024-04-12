import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

INPUT_FOLDER = 'results/'
OUTPUT_FOLDER = 'graphs/'

def extract_duration(filename):
  eventDuration = {}
  parseSection = 0
  with open(filename) as f:
    for newline in f:
      if '--------------------' in newline:
        parseSection += 1
      elif parseSection == 1:
        indicator, eventID, time = newline.split()
        if indicator == 't':
          eventDuration[int(eventID)] = float(time)

  return eventDuration

def plot_duration(configData):
  plt.figure(figsize=(12, 6))
  plt.cla()
  colors=list(mcolors.TABLEAU_COLORS.keys()) # maximum 10 events
  yVals = range(len(configData))

  sortedConfig = sorted(config)
  xStart = [0] * 5

  for eventID in range(5):
    xLen = [] * 5
    for y, config in enumerate(sortedConfig):
      xLen[y] = configData[config][eventID]
    plt.barh(y=yVals, left=xStart, width=xLen, height=0.5, color=colors[eventID])
    for y in len(sortedConfig):
      xStart[y] += xLen[y]

  plt.xlabel('Time (ms)')
  plt.yticks(range(len(configData.keys())), configData.keys())
  plt.legend(eventList.values(), loc='upper right', bbox_to_anchor=(1, 1))
  plt.savefig(OUTPUT_FOLDER + 'runtime.png', bbox_inches='tight')

if __name__ == '__main__':
  event_durations = {}
  for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith('.txt') and 'send' in filename:
      event_durations[filename] = extract_duration(INPUT_FOLDER + filename)
