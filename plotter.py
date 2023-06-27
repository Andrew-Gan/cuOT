import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

OUTPUT_FOLDER = 'output/'
hideEvents = ['AesKeyExpansion', 'MatrixInit', 'MatrixRand', 'MatrixMult', 'Hash']

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
        if eventString not in hideEvents:
          eventList[eventID] = eventString
          eventData[eventID] = []
      elif parseSection == 2:
        startStop, eventID, time = newline.split()
        eventID = int(eventID)
        if eventID not in eventList:
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
  plt.legend(eventList.values(), loc='lower left', bbox_to_anchor=(-.2, .25))

  plt.table([[eventList[i], f"{eventDurationSend[i]:.3f}"] for i in eventList],
    colWidths=[0.2, 0.15], colLabels=['Operation', 'Duration (ms)'],
    cellLoc='left', bbox=[1.01, 0, .25, .5])

  plt.table([[eventList[i], f"{eventDurationRecv[i]:.3f}"] for i in eventList],
    colWidths=[0.2, 0.15], colLabels=['Operation', 'Duration (ms)'],
    cellLoc='left', bbox=[1.01, .55, .25, .5])

  plt.savefig(OUTPUT_FOLDER + runconfig, bbox_inches='tight')

def plot_numtree_runtime(runconfig, eventList, eventDurationS, eventDurationsR, eventID):
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
  plt.plot(xVal, yValS)
  # plt.plot(xVal, yValR)
  plt.xscale('log', base=2)
  plt.legend(['Sender', 'Recver'])
  plt.title('Runtime of %s vs Number of PPRF Trees' % eventList[eventID])
  plt.savefig(OUTPUT_FOLDER + eventList[eventID], bbox_inches='tight')

def plot_custom_graph():
  xVal = list(range(7))
  expTime = [274, 253, 146, 150, 140, 137, 108]
  sumTime = [30, 20, 9, 9, 13, 14, 6]
  clusters = ['B', 'D', 'E', 'F', "G", 'H', 'K']
  gpu = ['A30', 'A30', 'V100', 'V100', 'A100', 'A10', 'A100']
  mem = [8, 8, 8, 16, 20, 8, 40]
  plt.bar([x-.05 for x in xVal], expTime, width=0.1)
  plt.bar([x+.05 for x in xVal], sumTime, width=0.1)
  plt.xticks(xVal, clusters)
  plt.table([[c, g, m, e, s, round(330/(e+s), 1)] for c, g, m, e, s in zip(clusters, gpu, mem, expTime, sumTime)],
    colLabels=['cluster', 'gpu', 'mem (GB)', 'exp (ms)', 'sum (ms)', 'speed up'],
    bbox=[0, -.9, 1, .75],
  )
  plt.xlabel('Sub Cluster')
  plt.ylabel('Runtime (ms)')
  plt.title('PPRF Expansion and Summation Runtime vs GPU Type')
  plt.savefig(OUTPUT_FOLDER + 'expansion-vs-cluster.png', bbox_inches='tight')

if __name__ == '__main__':
  # plot_custom_graph()

  runconfig = []
  eventDurationS = []
  eventDurationsR = []

  for filename in os.listdir(OUTPUT_FOLDER):
    if filename.endswith('send.txt'):
      src = '-'.join(filename.split('-')[:3])
      runconfig.append(src)
  runconfig = sorted(runconfig)

  config24 = []

  for src in runconfig:
    dataSend = extract_data(OUTPUT_FOLDER + src + '-send.txt')
    dataRecv = extract_data(OUTPUT_FOLDER + src + '-recv.txt')
    plot_pipeline(src, dataSend, dataRecv)
    if 'log-24' in src:
      config24.append(src)
      eventDurationS.append(dataSend[2])
      eventDurationsR.append(dataRecv[2])

  eventList = dataSend[0]
  plot_numtree_runtime(config24, eventList, eventDurationS, eventDurationsR, 2)
