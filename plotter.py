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
    nThread = 16

    plt.plot(inputSize, [dataPoints[i]['AES'][nThread-1] for i in inputSize])
    plt.plot(inputSize, [dataPoints[i]['AESNI'][nThread-1] for i in inputSize])
    plt.plot(inputSize, [dataPoints[i]['AESGPU'][nThread-1] for i in inputSize])

    plt.legend(['AES', 'AESNI', 'AESGPU'])
    plt.savefig('runtime.png')

def plot_exp(filename):
        runtimes = {}
        depths = set()
        x_inf_point = 0

        with open(filename, 'r') as f:
            nThread = 0
            for newline in f:
                if 'Depth' in newline:
                    depth = int(newline.split()[1][:-1])
                    nThread = int(newline.split()[-1])
                    depths.add(depth)
                    if nThread not in runtimes:
                        runtimes[nThread] = { 'aes': [], 'aesni': [], 'aesgpu': [] }
                elif 'AESNI' in newline:
                    runtimes[nThread]['aesni'].append(float(newline.split()[4]))
                elif 'AESGPU' in newline:
                    runtimes[nThread]['aesgpu'].append(float(newline.split()[4]))
                elif 'AES' in newline:
                    runtimes[nThread]['aes'].append(float(newline.split()[4]))

        depths = list(depths)
        depths.sort()

        plt.figure(figsize=(16, 32))

        for graph_idx, nThread in enumerate(runtimes):
            plt.subplot(len(runtimes), 2, 2 * graph_idx + 1)
            plt.title('Linear Runtime of Tree Expansion with Num Threads = %d' % nThread)
            plt.plot(depths, runtimes[nThread]['aes'], color='C1')
            plt.plot(depths, runtimes[nThread]['aesni'], color='C0')
            plt.plot(depths, runtimes[nThread]['aesgpu'], color='C2')
            plt.legend(['AES', 'AESNI', 'AESGPU'])
            plt.xticks(depths)
            plt.xlabel('Tree depth')
            plt.ylabel('Runtime (ms)')
            plt.xlim(15, 25)

            plt.subplot(len(runtimes), 2, 2 * graph_idx + 2)
            plt.title('Log Runtime of Tree Expansion with Num Threads = %d' % nThread)
            plt.plot(depths, runtimes[nThread]['aes'], color='C1')
            plt.plot(depths, runtimes[nThread]['aesni'], color='C0')
            plt.plot(depths, runtimes[nThread]['aesgpu'], color='C2')
            plt.legend(['AES', 'AESNI', 'AESGPU'])
            plt.xticks(depths)
            plt.yscale('log')
            plt.xlabel('Tree depth')
            plt.ylabel('Runtime (ms)')

        graph_idx += 2

        plt.tight_layout()

        plt.savefig('runtime.png')

if __name__ == '__main__':
    plot_exp('out')
