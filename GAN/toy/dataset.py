'''
Generate toy dataset
'''

import numpy as np

def toy_dataset(name):
    if name == '25Gaussian':
        dataset = []
        for i in range(100000 // 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828  # stdev
        return dataset

if __name__=='__main__':

    dataset = toy_dataset('25Gaussian')
    
    import matplotlib as mlp
    mlp.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.scatter(dataset[0:1000,0], dataset[0:1000,1], marker='.')
    plt.savefig('a.png')

    print(dataset.shape)
