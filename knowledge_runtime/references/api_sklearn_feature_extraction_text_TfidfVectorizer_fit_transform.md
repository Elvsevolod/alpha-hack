# sklearn.feature_extraction.text.TfidfVectorizer.fit_transform
Version: 1.5.2

## sklearn.feature_extraction.text.TfidfVectorizer.fit_transform(self, raw_documents, y=None)

Learn vocabulary and idf, return document-term matrix.

This is equivalent to fit followed by transform, but more efficiently
implemented.

Parameters
----------
raw_documents : iterable
    An iterable which generates either str, unicode or file objects.

y : None
    This parameter is ignored.

Returns
-------
X : sparse matrix of (n_samples, n_features)
    Tf-idf-weighted document-term matrix.
