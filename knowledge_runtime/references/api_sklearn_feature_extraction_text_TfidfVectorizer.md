# sklearn.feature_extraction.text.TfidfVectorizer

Version: 1.5.2

sklearn.feature_extraction.text.TfidfVectorizer(*, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer='word', stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<class 'numpy.float64'>, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)

Convert a collection of raw documents to a matrix of TF-IDF features.

Equivalent to :class:`CountVectorizer` followed by
:class:`TfidfTransformer`.

For an example of usage, see
:ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`.

For an efficiency comparison of the different feature extractors, see
:ref:`sphx_glr_auto_examples_text_plot_hashing_vs_dict_vectorizer.py`.

For an example of document clustering and comparison with
:class:`~sklearn.feature_extraction.text.HashingVectorizer`, see
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
    'unicode' is a slightly slower method that works on any characters.
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

stop_words : {'english'}, list, default=None
    If a string, it is passed to _check_stop_list and the appropriate stop
    list is returned. 'english' is currently the only supported string
    value.
    There are several known issues with 'english' and you should
    consider an alternative (see :ref:`stop_words`).

    If a list, that list is assumed to contain stop words, all of which
    will be removed from the resulting tokens.
    Only applies if ``analyzer == 'word'``.

    If None, no stop words will be used. In this case, setting `max_df`
    to a higher value, such as in the range (0.7, 1.0), can automatically detect
    and filter stop words based on intra corpus document frequency of terms.

token_pattern : str, default=r"(?u)\\b\\w\\w+\\b"
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

max_df : float or int, default=1.0
    When building the vocabulary ignore terms that have a document
    frequency strictly higher than the given threshold (corpus-specific
    stop words).
    If float in range [0.0, 1.0], the parameter represents a proportion of
    documents, integer absolute counts.
    This parameter is ignored if vocabulary is not None.

min_df : float or int, default=1
    When building the vocabulary ignore terms that have a document
    frequency strictly lower than the given threshold. This value is also
    called cut-off in the literature.
    If float in range of [0.0, 1.0], the parameter represents a proportion
    of documents, integer absolute counts.
    This parameter is ignored if vocabulary is not None.

max_features : int, default=None
    If not None, build a vocabulary that only consider the top
    `max_features` ordered by term frequency across the corpus.
    Otherwise, all features are used.

    This parameter is ignored if vocabulary is not None.

vocabulary : Mapping or iterable, default=None
    Either a Mapping (e.g., a dict) where keys are terms and values are
    indices in the feature matrix, or an iterable over terms. If not
    given, a vocabulary is determined from the input documents.

binary : bool, default=False
    If True, all non-zero term counts are set to 1. This does not mean
    outputs will have only 0/1 values, only that the tf term in tf-idf
    is binary. (Set `binary` to True, `use_idf` to False and
    `norm` to None to get 0/1 outputs).

dtype : dtype, default=float64
    Type of the matrix returned by fit_transform() or transform().

norm : {'l1', 'l2'} or None, default='l2'
    Each output row will have unit norm, either:

    - 'l2': Sum of squares of vector elements is 1. The cosine
      similarity between two vectors is their dot product when l2 norm has
      been applied.
    - 'l1': Sum of absolute values of vector elements is 1.
      See :func:`~sklearn.preprocessing.normalize`.
    - None: No normalization.

use_idf : bool, default=True
    Enable inverse-document-frequency reweighting. If False, idf(t) = 1.

smooth_idf : bool, default=True
    Smooth idf weights by adding one to document frequencies, as if an
    extra document was seen containing every term in the collection
    exactly once. Prevents zero divisions.

sublinear_tf : bool, default=False
    Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

Attributes
----------
vocabulary_ : dict
    A mapping of terms to feature indices.

fixed_vocabulary_ : bool
    True if a fixed vocabulary of term to indices mapping
    is provided by the user.

idf_ : array of shape (n_features,)
    The inverse document frequency (IDF) vector; only defined
    if ``use_idf`` is True.

See Also
--------
CountVectorizer : Transforms text into a sparse matrix of n-gram counts.

TfidfTransformer : Performs the TF-IDF transformation from a provided
    matrix of counts.

Examples
--------
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> vectorizer = TfidfVectorizer()
>>> X = vectorizer.fit_transform(corpus)
>>> vectorizer.get_feature_names_out()
array(['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third',
       'this'], ...)
>>> print(X.shape)
(4, 9)

