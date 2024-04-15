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

def plot_duration(event_durations):
  plt.figure(figsize=(12, 6))
  plt.cla()
  colors=list(mcolors.TABLEAU_COLORS.keys()) # maximum 10 events
  yVals = range(len(event_durations))

  xStart = [0] * len(event_durations)

  for eventID in range(1, 4):
    xLen = [0] * len(event_durations)
    for y, config in enumerate(event_durations):
      xLen[y] = event_durations[config][eventID]
    plt.barh(y=yVals, left=xStart, width=xLen, height=0.5, color=colors[eventID])
    for y in range(len(event_durations)):
      xStart[y] += xLen[y]

  plt.xlabel('Time (ms)')
  plt.yticks(range(len(event_durations.keys())), event_durations.keys())
  plt.legend(['BaseOT', 'SeedExp', 'LPN'], loc='upper right', bbox_to_anchor=(1, 1))
  plt.savefig(OUTPUT_FOLDER + 'runtime.png', bbox_inches='tight')

if __name__ == '__main__':
  event_durations = {}
  for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith('.txt') and 'gpu-ferret-send-25' in filename:
      ylabel = filename.split('-')[3] + '-' + filename.split('-')[4].split('.')[0]
      event_durations[ylabel] = extract_duration(INPUT_FOLDER + filename)
  plot_duration(event_durations)
