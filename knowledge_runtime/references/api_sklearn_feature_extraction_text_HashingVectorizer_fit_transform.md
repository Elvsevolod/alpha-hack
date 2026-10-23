# sklearn.feature_extraction.text.HashingVectorizer.fit_transform
Version: 1.5.2

## sklearn.feature_extraction.text.HashingVectorizer.fit_transform(self, X, y=None)

Transform a sequence of documents to a document-term matrix.

Parameters
----------
X : iterable over raw text documents, length = n_samples
    Samples. Each sample must be a text document (either bytes or
    unicode strings, file name or file object depending on the
    constructor argument) which will be tokenized and hashed.
y : any
    Ignored. This parameter exists only for compatibility with
    sklearn.pipeline.Pipeline.

Returns
-------
X : sparse matrix of shape (n_samples, n_features)
    Document-term matrix.
