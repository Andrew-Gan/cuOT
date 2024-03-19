import matplotlib.pyplot as plt

data = {}
x = range(15, 26)
legends = []

with open('result') as fp:
    newlines = fp.readlines()
    for newline in newlines:
        data[newline.split(':')[0]] = [float(v) for v in newline.split(':')[1].split(',')[:-1]]

for k, v in data.items():
    plt.plot(x, v)
    legends.append(k)

plt.legend(legends)
plt.title('Memcpy Runtime vs Copy Size')
plt.xlabel('Copy size (2^x bytes)')
plt.ylabel('Runtime (ms)')
plt.savefig('memcpy.png')
