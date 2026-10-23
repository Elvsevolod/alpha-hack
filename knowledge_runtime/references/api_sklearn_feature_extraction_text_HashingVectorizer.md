# sklearn.feature_extraction.text.HashingVectorizer

Version: 1.5.2

sklearn.feature_extraction.text.HashingVectorizer(*, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), analyzer='word', n_features=1048576, binary=False, norm='l2', alternate_sign=True, dtype=<class 'numpy.float64'>)

Convert a collection of text documents to a matrix of token occurrences.

It turns a collection of text documents into a scipy.sparse matrix holding
token occurrence counts (or binary occurrence information), possibly
normalized as token frequencies if norm='l1' or projected on the euclidean
unit sphere if norm='l2'.

This text vectorizer implementation uses the hashing trick to find the
token string name to feature integer index mapping.

This strategy has several advantages:

- it is very low memory scalable to large datasets as there is no need to
  store a vocabulary dictionary in memory.

- it is fast to pickle and un-pickle as it holds no state besides the
  constructor parameters.

- it can be used in a streaming (partial fit) or parallel pipeline as there
  is no state computed during fit.

There are also a couple of cons (vs using a CountVectorizer with an
in-memory vocabulary):

- there is no way to compute the inverse transform (from feature indices to
  string feature names) which can be a problem when trying to introspect
  which features are most important to a model.

- there can be collisions: distinct tokens can be mapped to the same
  feature index. However in practice this is rarely an issue if n_features
  is large enough (e.g. 2 ** 18 for text classification problems).

- no IDF weighting as this would render the transformer stateful.

The hash function employed is the signed 32-bit version of Murmurhash3.

For an efficiency comparison of the different feature extractors, see
:ref:`sphx_glr_auto_examples_text_plot_hashing_vs_dict_vectorizer.py`.

For an example of document clustering and comparison with
:class:`~sklearn.feature_extraction.text.TfidfVectorizer`, see
:ref:`sphx_glr_auto_examples_text_plot_document_clustering.py`.

Read more in the :ref:`User Guide <text_feature_extraction>`.

Parameters
----------
input : {'filename', 'file', 'content'}, default='content'
    - If `'filename'`, the sequence passed as an argument to fit is
      expected to be a list of filenames that need reading to fetch
      the raw content to analyze.

    - If `'file'`, the sequence items must have a 'read' method (file-like
      object) that is called to fetch the bytes in memory.

    - If `'content'`, the input is expected to be a sequence of items that
      can be of type string or byte.

encoding : str, default='utf-8'
    If bytes or files are given to analyze, this encoding is used to
    decode.

decode_error : {'strict', 'ignore', 'replace'}, default='strict'
    Instruction on what to do if a byte sequence is given to analyze that
    contains characters not of the given `encoding`. By default, it is
    'strict', meaning that a UnicodeDecodeError will be raised. Other
    values are 'ignore' and 'replace'.

strip_accents : {'ascii', 'unicode'} or callable, default=None
    Remove accents and perform other character normalization
    during the preprocessing step.
    'ascii' is a fast method that only works on characters that have
    a direct ASCII mapping.
    'unicode' is a slightly slower method that works on any character.
    None (default) means no character normalization is performed.

    Both 'ascii' and 'unicode' use NFKD normalization from
    :func:`unicodedata.normalize`.

lowercase : bool, default=True
    Convert all characters to lowercase before tokenizing.

preprocessor : callable, default=None
    Override the preprocessing (string transformation) stage while
    preserving the tokenizing and n-grams generation steps.
    Only applies if ``analyzer`` is not callable.

tokenizer : callable, default=None
    Override the string tokenization step while preserving the
    preprocessing and n-grams generation steps.
    Only applies if ``analyzer == 'word'``.

stop_words : {'english'}, list, default=None
    If 'english', a built-in stop word list for English is used.
    There are several known issues with 'english' and you should
    consider an alternative (see :ref:`stop_words`).

    If a list, that list is assumed to contain stop words, all of which
    will be removed from the resulting tokens.
    Only applies if ``analyzer == 'word'``.

token_pattern : str or None, default=r"(?u)\\b\\w\\w+\\b"
    Regular expression denoting what constitutes a "token", only used
    if ``analyzer == 'word'``. The default regexp selects tokens of 2
    or more alphanumeric characters (punctuation is completely ignored
    and always treated as a token separator).

    If there is a capturing group in token_pattern then the
    captured group content, not the entire match, becomes the token.
    At most one capturing group is permitted.

ngram_range : tuple (min_n, max_n), default=(1, 1)
    The lower and upper boundary of the range of n-values for different
    n-grams to be extracted. All values of n such that min_n <= n <= max_n
    will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
    unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
    only bigrams.
    Only applies if ``analyzer`` is not callable.

analyzer : {'word', 'char', 'char_wb'} or callable, default='word'
    Whether the feature should be made of word or character n-grams.
    Option 'char_wb' creates character n-grams only from text inside
    word boundaries; n-grams at the edges of words are padded with space.

    If a callable is passed it is used to extract the sequence of features
    out of the raw, unprocessed input.

    .. versionchanged:: 0.21
        Since v0.21, if ``input`` is ``'filename'`` or ``'file'``, the data
        is first read from the file and then passed to the given callable
        analyzer.

n_features : int, default=(2 ** 20)
    The number of features (columns) in the output matrices. Small numbers
    of features are likely to cause hash collisions, but large numbers
    will cause larger coefficient dimensions in linear learners.

binary : bool, default=False
    If True, all non zero counts are set to 1. This is useful for discrete
    probabilistic models that model binary events rather than integer
    counts.

norm : {'l1', 'l2'}, default='l2'
    Norm used to normalize term vectors. None for no normalization.

alternate_sign : bool, default=True
    When True, an alternating sign is added to the features as to
    approximately conserve the inner product in the hashed space even for
    small n_features. This approach is similar to sparse random projection.

    .. versionadded:: 0.19

dtype : type, default=np.float64
    Type of the matrix returned by fit_transform() or transform().

See Also
--------
CountVectorizer : Convert a collection of text documents to a matrix of
    token counts.
TfidfVectorizer : Convert a collection of raw documents to a matrix of
    TF-IDF features.

Notes
-----
This estimator is :term:`stateless` and does not need to be fitted.
However, we recommend to call :meth:`fit_transform` instead of
:meth:`transform`, as parameter validation is only performed in
:meth:`fit`.

Examples
--------
>>> from sklearn.feature_extraction.text import HashingVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> vectorizer = HashingVectorizer(n_features=2**4)
>>> X = vectorizer.fit_transform(corpus)
>>> print(X.shape)
(4, 16)

