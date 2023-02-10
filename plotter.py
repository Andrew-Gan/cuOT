import matplotlib.pyplot as plt

dataPoints = {}

with open('out', 'r') as f:
    lineCtr = 0
    newPoint = ''
    for newline in f:
        if lineCtr % 4 == 0:
            inputSize = int(newline.split(',')[0].split(' ')[1])
            nThreads = int(newline.split(',')[1].split(' ')[2])
            if inputSize not in dataPoints:
                dataPoints[inputSize] = { 'AES': [], 'AESNI': [], 'AESGPU': [] }
        else:
            dataType = newline.split(' ')[3][:-1]
            runTime = float(newline.split(' ')[4])
            dataPoints[inputSize][dataType].append(runTime)
        lineCtr += 1

inputSize = [2**i for i in range(15, 26)]
nThread = 8

plt.plot(inputSize, [dataPoints[i]['AES'][nThread-1] for i in inputSize])
plt.plot(inputSize, [dataPoints[i]['AESNI'][nThread-1] for i in inputSize])
plt.plot(inputSize, [dataPoints[i]['AESGPU'][nThread-1] for i in inputSize])

plt.legend(['AES', 'AESNI', 'AESGPU'])
plt.savefig('runtime.png')
