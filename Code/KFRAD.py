# Kernelized Fuzzy-Rough Anomaly Detection (KFRAD) algorithm
# Please refer to the following papers:
# Kernelized Fuzzy-Rough Anomaly Detection, IEEE Transactions on Fuzzy Systems, 2024.
# Uploaded by Yan Wu on Apr. 23, 2024. E-mail: wuyan7958@foxmail.com.
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler


def gaussian_matrix(Data, r):
    # input:
    # Data is data matrix without decisions, where rows for samples and columns for attributes.
    # Numerical attributes should be normalized into [0,1].
    # Nominal attributes be replaced by different integer values.
    # r is the Gaussian kernel parameter.

    n = Data.shape[0]
    m = Data.shape[1]
    transdata = np.zeros((n, m))
    transdata[:, 0:m] = Data
    temp = pdist(transdata, 'euclidean')  # calculate the Euclidean distance
    temp = squareform(temp)
    temp = np.exp(-(temp ** 2) / r)  # calculate the fuzzy similarity relation
    return temp


def KFRAD(data, delta):
    # input:
    # data is data matrix without decisions, where rows for samples and columns for attributes.
    # Numerical attributes should be normalized into [0,1].
    # Nominal attributes be replaced by different integer values.
    # delta is a given Gaussian kernel parameter for calculating the fuzzy similarity relation.

    # step1: Import data
    n, m = data.shape  # Number of rows and columns
    LA = np.arange(0, m)  # Attribute number 0~m-1
    weight1 = np.zeros((n, m))  # Attribute weights
    weight2 = np.zeros((n, m))  # Attribute weights

    Acc_A_a = np.zeros((n, m))  # Approximation accuracy
    for l in range(0, m):
        lA_d = np.setdiff1d(LA, l)  # The set of attributes after removing an attribute

        # step2: Compute the fuzzy relation matrix for a subset of single attributes
        NbrSet_tem = gaussian_matrix((np.matrix(data[:, l])).T, delta)
        NbrSet_temp, ia, ic = np.unique(NbrSet_tem, return_index=True, return_inverse=True, axis=0)

        for i in range(0, NbrSet_temp.shape[0]):
            # step3: Compute the fuzzy relation matrix for a subset of multiple attributes
            i_tem = np.where(ic == i)[0]
            data_tem = data[:, lA_d]
            NbrSet_tmp = gaussian_matrix(data_tem, delta)

            # step4: Compute the approximation accuracy

            # 4.1. Compute the fuzzy lower approximation
            a = 1 - NbrSet_tmp
            b = np.tile(NbrSet_temp[i, :], (n, 1))
            Low_A = sum((np.minimum(a + b - np.multiply(a, b) + np.multiply(np.sqrt(2 * a - np.multiply(a, a)),
                                                                            np.sqrt(2 * b - np.multiply(b, b))),
                                    1)).min(-1))

            # 4.2. Compute the fuzzy upper approximation
            a = NbrSet_tmp
            Upp_A = sum((np.maximum(
                np.multiply(a, b) - np.multiply(np.sqrt(1 - np.multiply(a, a)), np.sqrt(1 - np.multiply(b, b))),
                0)).max(-1))

            Acc_A_a[i_tem, l] = Low_A / Upp_A  # approximation accuracy

            # step5: Compute the corresponding weights
            weight2[i_tem, l] = 1 - (sum(NbrSet_temp[i, :]) / n) ** (1 / 3)
            weight1[i_tem, l] = (sum(NbrSet_temp[i, :]) / n)

    # step6: Compute the granule anomaly extent (GAE)
    GAE = np.zeros((n, m))
    for col in range(m):
        GAE[:, col] = 1 - (Acc_A_a[:, col]) * weight1[:, col]

    # step7: Compute the kernelized fuzzy rough anomaly score (KFRAS)
    KFRAS = np.array(np.mean(GAE * weight2, axis=1))

    return KFRAS


if __name__ == "__main__":
    load_data = loadmat('KFRAD_Example.mat')
    trandata = load_data['trandata']

    scaler = MinMaxScaler()
    trandata[:, :] = scaler.fit_transform(trandata[:, :])

    delta = 0.3
    out_scores = KFRAD(trandata, delta)

    print(out_scores)
