import sys
import matplotlib.pyplot as plt

hideEvents = ['PprfSender', 'PprfRecver', 'AesEncrypt', 'AesDecrypt',
  'AesInit', 'BaseOTInit']

def plot_pipeline(filename):
  eventList = {}
  eventData = {}
  tidFound = set()
  parseSection = 0
  with open(filename) as f:
    for newline in f:
      if '--------------------' in newline:
        parseSection += 1
      elif parseSection == 0:
        eventID, eventString = newline.split()
        eventList[int(eventID)] = eventString
        eventData[int(eventID)] = {}
      elif parseSection == 2:
        startStop, tid, eventID, time = newline.split()
        tid = int(tid)
        eventID = int(eventID)
        time = float(time)
        tidFound.add(tid)
        if tid not in eventData[eventID]:
          eventData[eventID][tid] = []
        if startStop == 's':
          eventData[eventID][tid].append([time, 0])
        elif startStop == 'e':
          startTime = eventData[eventID][tid][-1][0]
          eventData[eventID][tid][-1][1] = time - startTime

  sortedTids = sorted(list(tidFound))
  legends = []

  plt.figure(figsize=(12, 6))

  for eventID, eventVal in eventData.items():
    if len(eventVal) == 0 or eventList[eventID] in hideEvents:
      continue
    legends.append(eventList[eventID])
    eventTids = eventVal.keys()
    yTids = []
    widths = []
    starts = []
    for t in eventTids:
      for e in eventVal[t]:
        yTids.append(sortedTids.index(t))
        starts.append(e[0])
        widths.append(e[1])
    plt.barh(y=yTids, width=widths, height=0.5, left=starts)

  plt.yticks(range(len(tidFound)))
  plt.title('Pipeline Graph of Thread Operations over Time')
  plt.xlabel('Time (ms)')
  plt.ylabel('Thread ID')
  plt.legend(legends, loc='upper left', bbox_to_anchor=(1, 1))
  plt.savefig(filename.split('.')[0], bbox_inches='tight')

if __name__ == '__main__':
  plot_pipeline('log.txt')
