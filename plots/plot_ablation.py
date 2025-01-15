import matplotlib.pyplot as plt

basecolor=['indianred']
othercolors=['lightslategray', 'goldenrod', 'steelblue', 'darkseagreen']

titles = ['Seed Expansion', 'Dual LPN', 'Primal LPN']
filenames = ['ablation_seedexp.pdf', 'ablation_duallpn.pdf', 'ablation_primallpn.pdf']
yvals = [[
        [109, 19, 14, 5],
        [214.65, 26.72, 20.22, 8.28],
        [401.45, 39.99, 30.66, 15.55]
    ], [
        [3257.72, 891.98, 744.07, 720.89, 251.27],
        [7305.69, 1528.09, 1406.65, 1173.34, 502.153],
        [14998.90, 2614.80, 2259.53, 1866.12, 1016.53]
    ], [
        [52, 16.42, 14.28],
        [101, 32.29, 21.80],
        [141, 63.82, 55.21]
]]
xlabels = [
    ['CPU', 'OptAES', 'CoalescedTree', 'CoalescedSum'],
    ['CPU', 'cuFFT', 'BitTran', 'CompProd', 'Casting'],
    ['CPU', 'BitMult', 'IntegratedXOR'],
]
# hr = [6, 12]
legendfontSize=26
fontSize=20
labelfontSize=32
afontSize=24
figsize=(17,4)
figsize2=(17.5,4)
colwidth=0.14
otnum = ['$2^{23}$', '$2^{24}$', '$2^{25}$']


def plot_seedexp(xlab, t, ys, f):
    plt.figure(figsize=figsize)

    xvals = []
    for i, y in enumerate(ys):
        y = list(map(int, y))
        xloc = [i + j * 0.75 / len(y) for j in range(len(y))]
        bars = plt.bar(xloc, y, width=colwidth, color=basecolor+othercolors[5-len(y):])

        plt.legend(bars, xlab, ncol=4,  loc='upper center', fontsize=legendfontSize)

        for i, j in zip(xloc, y):
            plt.annotate(f'{j}', xy=(i, j), ha='center', va='bottom', fontsize=afontSize)

        xvals += xloc
    plt.xticks([i+0.075*len(y) for i in range(0, len(ys))],
        labels=otnum, fontsize=labelfontSize)
    plt.ylabel('Runtime (ms)', fontsize=labelfontSize)
    plt.yscale('log')
    plt.ylim(1, max(y)*100)
    plt.yticks(fontsize=labelfontSize)
    
    plt.tight_layout()
    plt.savefig(f'{f}')


def plot_dual(xlab, t, ys, f):
    plt.figure(figsize=figsize2)

    xvals = []
    xlocs = [0, 0.18, 0.36, 0.54, 0.72]
    for i, y in enumerate(ys):
        y = list(map(int, y))
        xloc = [i + j for j in xlocs]
        bars = plt.bar(xloc, y, width=colwidth, color=basecolor+othercolors[5-len(y):])
        plt.legend(bars, xlab, ncol=5, loc='upper center', fontsize=legendfontSize)

        for i, j in zip(xloc, y):
            plt.annotate(f'{j}', xy=(i, j), ha='center', va='bottom', fontsize=afontSize)

        xvals += xloc
    plt.xticks([i+0.075*len(y) for i in range(0, len(ys))],
        labels=otnum, fontsize=labelfontSize)
    plt.ylabel('Runtime (ms)', fontsize=labelfontSize)
    plt.yscale('log', base=10)
    plt.ylim(1, max(y)*200)
    plt.yticks(fontsize=labelfontSize)
    
    plt.tight_layout()
    plt.savefig(f'{f}')


def plot_primal(xlab, t, ys, f):
    plt.figure(figsize=figsize)

    xvals = []
    for i, y in enumerate(ys):
        y = list(map(int, y))
        xloc = [i + j * 0.55 / len(y) for j in range(len(y))]
        #bars = plt.bar(xloc, y, width=0.1, color=[f'C{i}' for i in range(len(y))])
        bars = plt.bar(xloc, y, width=colwidth, color=basecolor+othercolors[5-len(y):])
        plt.legend(bars, xlab,  ncol=3,  loc='upper center', fontsize=legendfontSize)

        for i, j in zip(xloc, y):
            plt.annotate(f'{j}', xy=(i, j), ha='center', va='bottom', fontsize=afontSize)

        xvals += xloc
    plt.xticks([i+0.075*len(y) for i in range(0, len(ys))],
        labels=otnum, fontsize=labelfontSize)
    plt.ylabel('Runtime (ms)', fontsize=labelfontSize)
    plt.yscale('log')
    plt.ylim(1, max(y)*20)
    plt.yticks(fontsize=labelfontSize)
    
    plt.tight_layout()
    plt.savefig(f'{f}')

idx=0
plot_seedexp(xlabels[idx], titles[idx], yvals[idx], filenames[idx])
idx=1
plot_dual(xlabels[idx], titles[idx], yvals[idx], filenames[idx])
idx=2
plot_primal(xlabels[idx], titles[idx], yvals[idx], filenames[idx])
