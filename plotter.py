import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

OUTPUT_FOLDER = 'results/'

def extract_data(filename):
  eventList = {}
  eventData = {}
  parseSection = 0
  with open(filename) as f:
    for newline in f:
      if '--------------------' in newline:
        parseSection += 1
      elif parseSection == 0:
        eventID, eventString = newline.split()
        eventID = int(eventID)
        eventList[eventID] = eventString
      elif parseSection == 1:
        startStop, eventID, time = newline.split()
        if startStop == 's' or startStop == 'e':
          eventID = int(eventID)
          time = float(time)
          if eventID not in eventData:
            eventData[eventID] = [[], []]
          if startStop == 's':
            eventData[eventID][0].append(time)
          elif startStop == 'e':
            startTime = eventData[eventID][0][-1]
            eventData[eventID][1].append(time - startTime)
  eventDuration = {}
  for eventID in eventList:
    eventDuration = {eventID: sum(eventData[eventID][1]) for eventID in eventData}

  return eventList, eventData, eventDuration

def plot_pipeline(eventList, configData):
  plt.figure(figsize=(12, 6))
  plt.cla()
  colors=list(mcolors.TABLEAU_COLORS.keys()) # maximum 10 events

  for eventID in eventList.keys():
    xStart = []
    xLen = []
    yVals = []
    for y, data in enumerate(configData.values()):
      if eventID not in data[0]:
        continue
      yVals += [y] * len(data[0][eventID][0])
      xStart += data[0][eventID][0]
      xLen += data[0][eventID][1]

    plt.barh(y=yVals, left=xStart, width=xLen, height=0.5, color=colors[eventID])

  plt.title('Pipeline Graph over Time')
  plt.xlabel('Time (ms)')
  plt.yticks(range(len(configData.keys())), configData.keys())
  plt.legend(eventList.values(), loc='lower left', bbox_to_anchor=(-.2, .4))
  plt.savefig(OUTPUT_FOLDER + 'pipeline.png', bbox_inches='tight')

def plot_numtree_runtime(runConfig, eventList, eventDuration, eventID):
  xVal = []
  yVal = []
  for run, durationSend in zip(runConfig, eventDuration):
    numTree = run.split('-')[3]
    xVal.append(int(numTree))
    yVal.append(durationSend[eventID])
  plt.cla()
  plt.figure(figsize=(12, 12))
  plt.plot(xVal, yVal)
  plt.xscale('log', base=2)
  plt.title('Runtime of Sender %s vs Number of PPRF Trees' % eventList[eventID])
  plt.savefig(OUTPUT_FOLDER + eventList[eventID], bbox_inches='tight')

if __name__ == '__main__':
  # plot_custom_graph()

  runConfig = []
  eventDuration = []
  for filename in os.listdir(OUTPUT_FOLDER):
    if filename.endswith('.txt'):
      src = filename.split('.')[0]
      runConfig.append(src)
  runConfig = sorted(runConfig)

  selectedConfig = []
  configData = {}

  eventList = []

  for filename in runConfig:
    ret = extract_data(OUTPUT_FOLDER + filename + '.txt')
    if len(eventList) == 0:
      eventList = ret[0]
    configData[filename] = ret[1:]

  plot_pipeline(eventList, configData)

  # eventList = configData[0][0]
  # plot_numtree_runtime(selectedConfig, eventList, eventDuration, 2)
