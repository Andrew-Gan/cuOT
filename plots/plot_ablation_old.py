import matplotlib.pyplot as plt

titles = ['Seed Expansion', 'Dual LPN', 'Primal LPN']
filenames = ['ablation_seedexp.png', 'ablation_duallpn.png', 'ablation_primallpn.png']
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
    ['CPU Baseline', 'Opt. AES', 'Coalesced Tree', 'Coalesced Sum'],
    ['CPU Baseline', 'CuFFT', 'Bit Trans.', 'Complex Prod.', 'Opt. Casting'],
    ['CPU Baseline', 'Bit Mult', 'Integrated XOR'],
]
# hr = [6, 12]
fontSize=20

for xlab, t, ys, f in zip(xlabels, titles, yvals, filenames):
    flat = [i for y in ys for i in y]
    # ranges = ((0, sorted(flat)[-2]*1.25), (max(flat)*0.9, max(flat)*1.5))
    # fig, ax = plt.subplots(1, 1, figsize=(12, 4), sharex=True)
    plt.figure(figsize=(15, 3))

    xvals = []
    for i, y in enumerate(ys):
        y = list(map(int, y))
        xloc = [i + j * 0.85 / len(y) for j in range(len(y))]
        bars = plt.bar(xloc, y, width=0.1, color=[f'C{i}' for i in range(len(y))])
        plt.legend(bars, xlab, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=fontSize)

        for i, j in zip(xloc, y):
            if t == 'Dual LPN':
                plt.annotate(f'{j}', xy=(i, j), ha='center', va='bottom', fontsize=fontSize-5)
            else:
                plt.annotate(f'{j}', xy=(i, j), ha='center', va='bottom', fontsize=fontSize+5)

        xvals += xloc
    plt.xticks([i+0.075*len(y) for i in range(0, len(ys))],
        labels=['2²³', '2²⁴', '2²⁵'], fontsize=fontSize+10)
    plt.ylabel('Runtime (ms)', fontsize=fontSize)
    plt.yscale('log')
    plt.ylim(1, max(y)*10)
    plt.yticks(fontsize=fontSize)
    
    # d = .5  # proportion of vertical to horizontal extent of the slanted line
    # kwargs = dict(marker=[(-1, -d), (1, d)], markersize=4,
    #             linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    # plt.ylabel('Runtime (ms)', loc='top', fontsize=fontSize)

    # # zoom-in / limit the view to different portions of the data
    # for i, [ax, rang] in enumerate(zip(axs, ranges)):
    #     ax.set_ylim(rang)
    #     ax.tick_params(bottom=False)
    #     if i != 0:
    #         ax.spines.bottom.set_visible(False)
    #         ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)
    #     if i != len(axs)-1:
    #         ax.spines.top.set_visible(False)
    #         ax.plot([0, 1], [1, 1], transform=ax.transAxes, **kwargs)

    plt.tight_layout()
    plt.savefig(f'{f}')
