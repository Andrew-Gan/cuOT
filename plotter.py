import sys
import matplotlib.pyplot as plt

hideEvents = ['PprfSender', 'PprfRecver']

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
          eventData[eventID][tid] = [0, 0]
        if startStop == 's':
          eventData[eventID][tid][0] = time
        elif startStop == 'e':
          startTime = eventData[eventID][tid][0]
          eventData[eventID][tid][1] = time-startTime

  tidFound = sorted(list(tidFound))
  legends = []

  plt.figure(figsize=(10, 5))

  for eventID, eventVal in eventData.items():
    if len(eventVal) == 0 or eventList[eventID] in hideEvents:
      continue
    legends.append(eventList[eventID])
    originalTids = eventVal.keys()
    widths = [eventVal[t][1] for t in originalTids]
    starts = [eventVal[t][0] for t in originalTids]
    mappedTids = [tidFound.index(tid) for tid in eventVal.keys()]
    plt.barh(y=mappedTids, width=widths, height=0.5, left=starts)

  plt.legend(legends)
  plt.savefig(filename.split('.')[0])

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('usage: python plotter <logfile>')
  plot_pipeline(sys.argv[1])
