import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

OUTPUT_FOLDER = 'output/'
hideEvents = []

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
        if eventList[eventID] not in hideEvents:
          eventData[eventID] = []
      elif parseSection == 2:
        startStop, eventID, time = newline.split()
        eventID = int(eventID)
        if eventList[eventID] in hideEvents:
          continue
        time = float(time)
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

def plot_pipeline(runconfig, dataSend, dataRecv):
  eventList, eventDataSend, eventDurationSend = dataSend
  eventList, eventDataRecv, eventDurationRecv = dataRecv
  logOT = runconfig.split('-')[1]
  numTree = runconfig.split('-')[2]
  plt.figure(figsize=(12, 6))
  plt.cla()
  colors=list(mcolors.TABLEAU_COLORS.keys()) # maximum 10 events
  for i, eventData in enumerate([eventDataSend, eventDataRecv]):
    for colorCode, [eventID, eventVal] in zip(colors, eventData.items()):
      widths = []
      starts = []
      for e in eventVal:
        starts.append(e[0])
        widths.append(e[1])
      plt.barh(y=[i], width=widths, height=0.5, left=starts, color=colorCode)
  plt.title('Pipeline Graph over Time with n=%s and t=%s' % (logOT, numTree))
  plt.xlabel('Time (ms)')
  plt.yticks([0, 1], ['Sender', 'Recver'])
  plt.legend(eventList.values(), loc='lower left', bbox_to_anchor=(-.2, 0.25))

  plt.table([[eventList[i], f"{eventDurationSend[i]:.3f}"] for i in eventList],
    colWidths=[0.2, 0.15], colLabels=['Operation', 'Duration (ms)'],
    cellLoc='left', bbox=[1.01, 0, .25, .5])

  plt.table([[eventList[i], f"{eventDurationRecv[i]:.3f}"] for i in eventList],
    colWidths=[0.2, 0.15], colLabels=['Operation', 'Duration (ms)'],
    cellLoc='left', bbox=[1.01, .55, .25, .5])

  plt.savefig(OUTPUT_FOLDER + runconfig, bbox_inches='tight')

def plot_tree_graph(runconfig, eventList, eventDurationS, eventDurationsR, eventID):
  xVal = []
  yValS = []
  yValR = []
  for run, durationSend, durationRecv in zip(runconfig, eventDurationS, eventDurationsR):
    numTree = run.split('-')[2].split('.')[0]
    xVal.append(int(numTree))
    yValS.append(durationSend[eventID])
    yValR.append(durationRecv[eventID])
  plt.figure(figsize=(12, 6))
  plt.cla()
  plt.scatter(xVal, yValS)
  plt.scatter(xVal, yValR)
  plt.title('Runtime of %s vs Number of PPRF Trees' % eventList[eventID])
  plt.savefig(OUTPUT_FOLDER + eventList[eventID], bbox_inches='tight')

if __name__ == '__main__':
  runconfig = []
  eventDurationS = []
  eventDurationsR = []

  for filename in os.listdir(OUTPUT_FOLDER):
    if filename.endswith('send.txt'):
      src = '-'.join(filename.split('-')[:3])
      runconfig.append(src)

  for src in runconfig:
    dataSend = extract_data(OUTPUT_FOLDER + src + '-send.txt')
    dataRecv = extract_data(OUTPUT_FOLDER + src + '-recv.txt')
    plot_pipeline(src, dataSend, dataRecv)
    eventDurationS.append(dataSend[2])
    eventDurationsR.append(dataRecv[2])

  eventList = dataSend[0]
  plot_tree_graph(runconfig, eventList, eventDurationS, eventDurationsR, 3)
