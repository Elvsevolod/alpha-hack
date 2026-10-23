# sklearn.feature_extraction.text.TfidfVectorizer.transform
Version: 1.5.2

## sklearn.feature_extraction.text.TfidfVectorizer.transform(self, raw_documents)

Transform documents to document-term matrix.

Uses the vocabulary and document frequencies (df) learned by fit (or
fit_transform).

Parameters
----------
raw_documents : iterable
    An iterable which generates either str, unicode or file objects.

Returns
-------
X : sparse matrix of (n_samples, n_features)
    Tf-idf-weighted document-term matrix.
