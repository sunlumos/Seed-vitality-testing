import numpy as np
import pandas as pd


def ks(x, y, test_size=2/3):
    """
    :param x: shape (n_samples, n_features)
    :param y: shape (n_sample, )
    :param test_size: the ratio of test_size (float)
    :return: spec_train: (n_samples, n_features)
             spec_test: (n_samples, n_features)
             target_train: (n_sample, )
             target_test: (n_sample, )
    """
    M = x.shape[0]
    N = round((1 - test_size) * M)
    samples = np.arange(M)

    D = np.zeros((M, M))

    for i in range((M - 1)):
        xa = x[i, :]
        for j in range((i + 1), M):
            xb = x[j, :]
            D[i, j] = np.linalg.norm(xa - xb)

    maxD = np.max(D, axis=0)
    index_row = np.argmax(D, axis=0)
    index_column = np.argmax(maxD)

    m = np.zeros(N)
    m[0] = np.array(index_row[index_column])
    m[1] = np.array(index_column)
    m = m.astype(int)
    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]

    for i in range(2, N):
        pool = np.delete(samples, m[:i])
        dmin = np.zeros((M - i))
        for j in range((M - i)):
            indexa = pool[j]
            d = np.zeros(i)
            for k in range(i):
                indexb = m[k]
                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)
        dminmax[i] = np.max(dmin)
        index = np.argmax(dmin)
        m[i] = pool[index]

    m_complement = np.delete(np.arange(x.shape[0]), m)

    spec_train = x[m, :]
    # target_train = y[m]
    spec_test = x[m_complement, :]
    # target_test = y[m_complement]
    return spec_train, spec_test,
fi =  pd.read_csv("D:\S\start\code\seeds\Triple classification data\longjingyou_weilaohua.csv", header=None)
fi = np.array(fi)
spec_train, spec_test, = ks(fi,600)
np.savetxt('longjingyou_weilaohua_train.csv',spec_test,fmt='%.4f',delimiter=',')
np.savetxt('longjingyou_weilaohua_val+test.csv',spec_train,fmt='%.4f',delimiter=',')