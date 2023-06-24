import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

hideEvents = ['MatrixInit']

def plot_pipeline(filename):
  logOT = filename.split('-')[1]
  numTree = filename.split('-')[2].split('.')[0]

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
        eventID = int(eventID)
        eventList[eventID] = eventString
        if eventList[eventID] not in hideEvents:
          eventData[eventID] = {}
      elif parseSection == 2:
        startStop, tid, eventID, time = newline.split()
        tid = int(tid)
        eventID = int(eventID)
        if eventList[eventID] in hideEvents:
          continue
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
  colors=list(mcolors.TABLEAU_COLORS.keys()) # maximum 10 events

  for colorCode, [eventID, eventVal] in zip(colors, eventData.items()):
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
    plt.barh(y=yTids, width=widths, height=0.5, left=starts, color=colorCode)

  plt.yticks(range(len(tidFound)))
  plt.title('Pipeline of Thread Operations over Time with n=%s and t=%s' % (logOT, numTree))
  plt.xlabel('Time (ms)')
  plt.ylabel('Thread ID')
  plt.legend(legends, loc='upper right', bbox_to_anchor=(1, 1))

  eventDuration = {}
  for eventID in eventList:
    eventDuration[eventID] = 0
    if eventID in eventData:
      for tid in eventData[eventID].values():
        for event in tid:
          eventDuration[eventID] += event[1]

  plt.table([[eventList[i], f"{eventDuration[i]:.3f}"] for i in eventList],
    colWidths=[0.2, 0.15], colLabels=['Operation', 'Duration (ms)'],
    cellLoc='left', loc='bottom right')

  plt.savefig(filename.split('.')[0], bbox_inches='tight')

# def plot_numtree_vs_runtime(filename):


if __name__ == '__main__':
  plot_pipeline(sys.argv[1])
