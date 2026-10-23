# sklearn.feature_extraction.text.HashingVectorizer.transform
Version: 1.5.2

## sklearn.feature_extraction.text.HashingVectorizer.transform(self, X)

Transform a sequence of documents to a document-term matrix.

Parameters
----------
X : iterable over raw text documents, length = n_samples
    Samples. Each sample must be a text document (either bytes or
    unicode strings, file name or file object depending on the
    constructor argument) which will be tokenized and hashed.

Returns
-------
X : sparse matrix of shape (n_samples, n_features)
    Document-term matrix.
