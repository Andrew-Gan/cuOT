import matplotlib.pyplot as plt

def plot_enc(filename):
    dataPoints = {}

    with open(filename, 'r') as f:
        lineCtr = 0
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

def plot_exp(filename):
    depth = []
    aesRuntime = []
    aesniRuntime = []
    aesgpuRuntime = []

    x_inf_point = 0

    with open(filename, 'r') as f:
        for newline in f:
            if 'Depth' in newline:
                depth.append(int(newline.split()[1][:-1]))
            elif 'AESNI' in newline:
                aesniRuntime.append(float(newline.split()[4]))
            elif 'AESGPU' in newline:
                aesgpuRuntime.append(float(newline.split()[4]))
                if aesniRuntime[-1] <= aesgpuRuntime[-1]:
                    x_inf_point = depth[-1]
            elif 'AES' in newline:
                aesRuntime.append(float(newline.split()[4]))

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(depth, aesRuntime, color='C1')
    plt.plot(depth, aesniRuntime, color='C0')
    plt.plot(depth, aesgpuRuntime, color='C2')
    plt.legend(['AESNI', 'AESGPU'])
    plt.legend(['AES', 'AESNI', 'AESGPU'])
    plt.title('Runtime for GMM Tree Expansion over Different Depths')
    plt.xticks(depth)
    plt.xlabel('Tree depth')
    plt.ylabel('Runtime (ms)')

    plt.subplot(2, 1, 2)
    plt.plot(depth, aesRuntime, color='C1')
    plt.plot(depth, aesniRuntime, color='C0')
    plt.plot(depth, aesgpuRuntime, color='C2')
    plt.legend(['AESNI', 'AESGPU'])
    plt.legend(['AES', 'AESNI', 'AESGPU'])
    plt.title('Runtime for GMM Tree Expansion over Different Depths')
    plt.xticks(depth)
    plt.yscale('log')
    plt.xlabel('Tree depth')
    plt.ylabel('Runtime (ms)')

    plt.tight_layout()

    plt.savefig('runtime.png')

if __name__ == '__main__':
    plot_exp('out')
