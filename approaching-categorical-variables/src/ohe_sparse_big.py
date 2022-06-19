# This script also demonstrates the different in memory storage
# needed when working with OHE data representation compared to 
# using the sparse representation.

# The results show that simply using OHE dense representation
# for this sample data would needs 8GB of memory, while the 
# sparse representation needs 8MB of memory. 

import numpy as np
from sklearn import preprocessing

if __name__ == "__main__":

    # create a 1-D array with 1001 different categories (int)
    example = np.random.randint(1000, size=1000000)

    # initialize OneHotEncoder from sklearn without converting to sparse rep.
    ohe = preprocessing.OneHotEncoder(sparse=False)

    # fit and transform data with one hot encoder
    ohe_example = ohe.fit_transform(example.reshape(-1, 1))

    # print size in bytes of dense array
    print(f"Size of dense array = {ohe_example.nbytes}")

    # initialize OneHotEncoder from sklearn with converting to sparse rep.
    ohe = preprocessing.OneHotEncoder(sparse=True)

    # fit and transform data with one hot encoder
    ohe_example = ohe.fit_transform(example.reshape(-1, 1))

    # print size in bytes of dense array
    print(f"Size of sparse array = {ohe_example.data.nbytes}")

    full_size = (
        ohe_example.data.nbytes +
        ohe_example.indptr.nbytes + ohe_example.indices.nbytes
    )

    # print full size of this sparse matrix
    print(f"Full size of sparse array = {full_size}")