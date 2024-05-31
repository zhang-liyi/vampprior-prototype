import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_images(args, x_sample, filename='img.pdf', size_x=3, size_y=3):
    # x_sample has shape (num_datapoints, 784)

    fig = plt.figure(figsize=(size_x, size_y))
    # fig = plt.figure(1)
    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(x_sample):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        sample = sample.reshape((args.input_size[0], args.input_size[1], args.input_size[2]))
        sample = sample.swapaxes(0, 2)
        sample = sample.swapaxes(0, 1)
        if args.input_type == 'binary' or args.input_type == 'gray':
            sample = sample[:, :, 0]
            plt.imshow(sample, cmap='gray')
        else:
            plt.imshow(sample)
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_kmeans(scores_standard, scores_vampprior, k_list):

    plt.rcParams['figure.figsize'] = [5,4]
    plt.rcParams['axes.linewidth'] = 0.1
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.plot(k_list, scores_standard, label='standard', c='orange', linewidth=3)
    plt.plot(k_list, scores_vampprior, label='vampprior', c='purple', linewidth=3)
    plt.legend()
    plt.savefig('kmeans_mnist' + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()