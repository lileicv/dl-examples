import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch

def generate_image(true_dist, netG, netD, savepath):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3

    # 生成判别器决策面
    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = torch.tensor(points.reshape((-1, 2))).cuda()
    disc_map = netD(points).cpu().data.numpy()
    
    # 生成样本
    noise = torch.randn(1000, 2).cuda()
    samples = netG(noise).cpu().data.numpy()

    plt.figure()
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose(), zorder=0)

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='red', marker='+', zorder=1, alpha=0.1)
    plt.scatter(samples[:, 0], samples[:, 1], c='blue', marker='+', zorder=2, alpha=0.1)

    plt.savefig(savepath)
    plt.close()
