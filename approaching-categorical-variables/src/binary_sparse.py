# This script shows how much much memory a simple dataset
# of 3 samples and 3 different features uses when represented
# in binary representation, and also how much memory is used
# when the binary representation is converted to a sparse representation

import numpy as np
from scipy import sparse

if __name__ == "__main__":

    # binary representation of data
    example = np.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 1]
        ]
    )

    # get number of bytes
    binary_memory = example.nbytes

    # change the binary representation to a sparse representation
    # and calculate total number of memory used
    sparse_example = sparse.csr_matrix(example)
    sparse_memory = sparse_example.data.nbytes + sparse_example.indptr.nbytes + sparse_example.indices.nbytes
    
    print(f"The binary representation uses {binary_memory} bytes of storage.")
    print(f"The sparse representation uses {sparse_memory} bytes of storage.")