# scipy.sparse.hstack

Version: 1.12.0

scipy.sparse.hstack(blocks, format=None, dtype=None)

Stack sparse matrices horizontally (column wise)

Parameters
----------
blocks
    sequence of sparse matrices with compatible shapes
format : str
    sparse format of the result (e.g., "csr")
    by default an appropriate sparse matrix format is returned.
    This choice is subject to change.
dtype : dtype, optional
    The data-type of the output matrix. If not given, the dtype is
    determined from that of `blocks`.

Returns
-------
new_array : sparse matrix or array
    If any block in blocks is a sparse array, return a sparse array.
    Otherwise return a sparse matrix.

    If you want a sparse array built from blocks that are not sparse
    arrays, use `block(hstack(blocks))` or convert one block
    e.g. `blocks[0] = csr_array(blocks[0])`.

See Also
--------
vstack : stack sparse matrices vertically (row wise)

Examples
--------
>>> from scipy.sparse import coo_matrix, hstack
>>> A = coo_matrix([[1, 2], [3, 4]])
>>> B = coo_matrix([[5], [6]])
>>> hstack([A,B]).toarray()
array([[1, 2, 5],
       [3, 4, 6]])
