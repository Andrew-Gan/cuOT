import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

OUTPUT_FOLDER = 'output/'
hideEvents = []

def extract_data(filename):
  eventIdToStr = {}
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
        if eventString not in hideEvents:
          eventIdToStr[eventID] = eventString
      elif parseSection == 1:
        startStop, eventID, time = newline.split()
        if startStop == 's' or startStop == 'e':
          eventID = int(eventID)
          eventList[eventID] = eventIdToStr[eventID]
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

def plot_pipeline(runConfig, configData):
  plt.figure(figsize=(12, 6))
  plt.cla()
  colors=list(mcolors.TABLEAU_COLORS.keys()) # maximum 10 events

  for conf, data in zip(runConfig, configData):
    eventList, eventData, eventDuration = data
    logOT = conf.split('-')[2]
    numTree = conf.split('-')[3]
    yVal = 0
    if 'send' in conf:
      yVal += 1
    if 'cpu' in conf:
      yVal += 2
    for colorCode, [eventID, eventVal] in zip(colors, eventData.items()):
      widths = []
      starts = []
      for e in eventVal:
        starts.append(e[0])
        widths.append(e[1])
      plt.barh(y=[yVal], width=widths, height=0.5, left=starts, color=colorCode)

    plt.table([[eventList[i], f"{eventDuration[i]:.3f}"] for i in eventList],
    colWidths=[0.2, 0.15], colLabels=['Operation', 'Duration (ms)'],
    cellLoc='left', bbox=[1.01, 0.25 * yVal, .25, .2])

  plt.title('Pipeline Graph over Time with n=%s and t=%s' % (logOT, numTree))
  plt.xlabel('Time (ms)')
  plt.yticks(range(4), ['GPU Recver', 'GPU Sender', 'CPU Recver', 'CPU Sender'])
  plt.legend(eventList.values(), loc='lower left', bbox_to_anchor=(-.2, .4))
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

# def plot_custom_graph():
#   xVal = list(range(7))
#   expTime = [274, 253, 146, 150, 140, 137, 108]
#   sumTime = [30, 20, 9, 9, 13, 14, 6]
#   clusters = ['B', 'D', 'E', 'F', "G", 'H', 'K']
#   gpu = ['A30', 'A30', 'V100', 'V100', 'A100', 'A10', 'A100']
#   mem = [8, 8, 8, 16, 20, 8, 40]
#   plt.bar([x-.05 for x in xVal], expTime, width=0.1)
#   plt.bar([x+.05 for x in xVal], sumTime, width=0.1)
#   plt.xticks(xVal, clusters)
#   plt.table([[c, g, m, e, s, round(330/(e+s), 1)] for c, g, m, e, s in zip(clusters, gpu, mem, expTime, sumTime)],
#     colLabels=['cluster', 'gpu', 'mem (GB)', 'exp (ms)', 'sum (ms)', 'speed up'],
#     bbox=[0, -.9, 1, .75],
#   )
#   plt.xlabel('Sub Cluster')
#   plt.ylabel('Runtime (ms)')
#   plt.title('PPRF Expansion and Summation Runtime vs GPU Type')
#   plt.savefig(OUTPUT_FOLDER + 'expansion-vs-cluster.png', bbox_inches='tight')

if __name__ == '__main__':
  # plot_custom_graph()

  runConfig = []
  eventDuration = []
  for filename in os.listdir(OUTPUT_FOLDER):
    if filename.endswith('.txt'):
      src = filename.split('.')[0]
      runConfig.append(src)
  runConfig = sorted(runConfig)

  config24 = []
  configData = []

  for src in runConfig:
    configData.append(extract_data(OUTPUT_FOLDER + src + '.txt'))
    if 'log-024' in src and 'send' in src:
      config24.append(src)
      eventDuration.append(configData[-1][2])

  plot_pipeline(runConfig, configData)

  eventList = configData[0][0]
  plot_numtree_runtime(config24, eventList, eventDuration, 2)
