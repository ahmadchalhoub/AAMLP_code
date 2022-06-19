# This script shows how much much memory a simple dataset
# of 3 samples and 3 different features uses when represented
# in one hot encoding representation, and also how much memory is used
# when the one hot encodoing representation is converted to a sparse representation.

# Refer to pages 94, 95, and 96 from the book for more details.

import numpy as np
from scipy import sparse

if __name__ == "__main__":

    # one hot encoding representation of data
    example = np.array(
        [
            [0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0]
        ]
    )

    # get number of bytes
    one_hot_encoding_memory = example.nbytes
    
    # change the one hot encoding representation to a sparse 
    # representation and calculate total number of memory used
    sparse_example = sparse.csr_matrix(example)
    sparse_memory = sparse_example.data.nbytes + sparse_example.indptr.nbytes + sparse_example.indices.nbytes
    
    print(f"The one hot encoding representation uses {one_hot_encoding_memory} bytes of storage.")
    print(f"The sparse representation uses {sparse_memory} bytes of storage.")