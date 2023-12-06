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
            eventData[eventID] = []
          if startStop == 's':
            eventData[eventID].append([time, 0])
          elif startStop == 'e':
            startTime = eventData[eventID][-1][0]
            eventData[eventID][-1][1] = time - startTime
  eventDuration = {}
  for eventID in eventList:
    eventDuration[eventID] = 0
    if eventID in eventData:
      for event in eventData[eventID]:
          eventDuration[eventID] += event[1]

  return eventList, eventData, eventDuration

def plot_pipeline(configData):
  plt.figure(figsize=(12, 6))
  plt.cla()
  colors=list(mcolors.TABLEAU_COLORS.keys()) # maximum 10 events

  for i, data in enumerate(configData.values()):
    eventList, eventData, eventDuration = data
    # logOT = conf.split('-')[2]
    # numTree = conf.split('-')[3]
    for eventKey, eventRange in eventData.items():
      widths = []
      starts = []
      for e in eventRange:
        starts.append(e[0])
        widths.append(e[1])
      plt.barh(y=i, width=widths, height=0.5, left=starts, color=colors[eventKey])

    plt.table([[eventList[i], f"{eventDuration[i]:.3f}"] for i in eventList],
    colWidths=[0.2, 0.15], colLabels=['Operation', 'Duration (ms)'],
    cellLoc='left', bbox=[1.01, 0.25 * i, .25, .2])

  plt.title('Pipeline Graph over Time')
  # plt.xscale('log')
  plt.xlabel('Time (ms)')
  plt.yticks(range(len(configData.keys())), configData.keys())
  plt.legend(configData['cpu-silent-send'][0].values(), loc='lower left', bbox_to_anchor=(-.2, .4))
  plt.savefig(OUTPUT_FOLDER + 'pipeline.png', bbox_inches='tight')

def plot_numtree_runtime(runConfig, eventList, eventDuration, eventID):
  xVal = []
  yVal = []
  for run, durationSend in zip(runConfig, eventDuration):
    numTree = run.split('-')[3]
    xVal.append(int(numTree))
    yVal.append(durationSend[eventID])
  plt.figure(figsize=(12, 6))
  plt.cla()
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

  for filename in runConfig:
    configData[filename] = extract_data(OUTPUT_FOLDER + filename + '.txt')

  plot_pipeline(configData)

  # eventList = configData[0][0]
  # plot_numtree_runtime(selectedConfig, eventList, eventDuration, 2)
