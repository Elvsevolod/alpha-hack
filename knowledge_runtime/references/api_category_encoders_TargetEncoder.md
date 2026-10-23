# category_encoders.TargetEncoder

Version: 2.6.4

category_encoders.TargetEncoder(verbose=0, cols=None, drop_invariant=False, return_df=True, handle_missing='value', handle_unknown='value', min_samples_leaf=20, smoothing=10, hierarchy=None)

Target encoding for categorical features.

Supported targets: binomial and continuous. For polynomial target support, see PolynomialWrapper.

For the case of categorical target: features are replaced with a blend of posterior probability of the target
given particular categorical value and the prior probability of the target over all the training data.

For the case of continuous target: features are replaced with a blend of the expected value of the target
given particular categorical value and the expected value of the target over all the training data.

Parameters
----------

verbose: int
    integer indicating verbosity of the output. 0 for none.
cols: list
    a list of columns to encode, if None, all string columns will be encoded.
drop_invariant: bool
    boolean for whether or not to drop columns with 0 variance.
return_df: bool
    boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
handle_missing: str
    options are 'error', 'return_nan'  and 'value', defaults to 'value', which returns the target mean.
handle_unknown: str
    options are 'error', 'return_nan' and 'value', defaults to 'value', which returns the target mean.
min_samples_leaf: int
    For regularization the weighted average between category mean and global mean is taken. The weight is
    an S-shaped curve between 0 and 1 with the number of samples for a category on the x-axis.
    The curve reaches 0.5 at min_samples_leaf. (parameter k in the original paper)
smoothing: float
    smoothing effect to balance categorical average vs prior. Higher value means stronger regularization.
    The value must be strictly bigger than 0. Higher values mean a flatter S-curve (see min_samples_leaf).
hierarchy: dict or dataframe
    A dictionary or a dataframe to define the hierarchy for mapping.

    If a dictionary, this contains a dict of columns to map into hierarchies.  Dictionary key(s) should be the column name from X
    which requires mapping.  For multiple hierarchical maps, this should be a dictionary of dictionaries.

    If dataframe: a dataframe defining columns to be used for the hierarchies.  Column names must take the form:
        HIER_colA_1, ... HIER_colA_N, HIER_colB_1, ... HIER_colB_M, ...
    where [colA, colB, ...] are given columns in cols list.  
    1:N and 1:M define the hierarchy for each column where 1 is the highest hierarchy (top of the tree).  A single column or multiple 
    can be used, as relevant.

Examples
-------
>>> from category_encoders import *
>>> import pandas as pd
>>> from sklearn.datasets import fetch_openml
>>> display_cols = ["Id", "MSSubClass", "MSZoning", "LotFrontage", "YearBuilt", "Heating", "CentralAir"]
>>> bunch = fetch_openml(name="house_prices", as_frame=True)
>>> y = bunch.target > 200000
>>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
>>> enc = TargetEncoder(cols=['CentralAir', 'Heating'], min_samples_leaf=20, smoothing=10).fit(X, y)
>>> numeric_dataset = enc.transform(X)
>>> print(numeric_dataset.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 7 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   Id           1460 non-null   float64
 1   MSSubClass   1460 non-null   float64
 2   MSZoning     1460 non-null   object 
 3   LotFrontage  1201 non-null   float64
 4   YearBuilt    1460 non-null   float64
 5   Heating      1460 non-null   float64
 6   CentralAir   1460 non-null   float64
dtypes: float64(6), object(1)
memory usage: 80.0+ KB
None

>>> from category_encoders.datasets import load_compass
>>> X, y = load_compass()
>>> hierarchical_map = {'compass': {'N': ('N', 'NE'), 'S': ('S', 'SE'), 'W': 'W'}}
>>> enc = TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=hierarchical_map, cols=['compass']).fit(X.loc[:,['compass']], y)
>>> hierarchy_dataset = enc.transform(X.loc[:,['compass']])
>>> print(hierarchy_dataset['compass'].values)
[0.62263617 0.62263617 0.90382995 0.90382995 0.90382995 0.17660024
 0.17660024 0.46051953 0.46051953 0.46051953 0.46051953 0.40332791
 0.40332791 0.40332791 0.40332791 0.40332791]
>>> X, y = load_postcodes('binary')
>>> cols = ['postcode']
>>> HIER_cols = ['HIER_postcode_1','HIER_postcode_2','HIER_postcode_3','HIER_postcode_4']
>>> enc = TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=X[HIER_cols], cols=['postcode']).fit(X['postcode'], y)
>>> hierarchy_dataset = enc.transform(X['postcode'])
>>> print(hierarchy_dataset.loc[0:10, 'postcode'].values)
[0.75063473 0.90208756 0.88328833 0.77041254 0.68891504 0.85012847
0.76772574 0.88742357 0.7933824  0.63776756 0.9019973 ]

References
----------

.. [1] A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems, from
https://dl.acm.org/citation.cfm?id=507538

Constructor parameters:
Parameters
----------

verbose: int
    integer indicating verbosity of output. 0 for none.
cols: list
    a list of columns to encode, if None, all string and categorical columns
    will be encoded.
drop_invariant: bool
    boolean for whether or not to drop columns with 0 variance.
return_df: bool
    boolean for whether to return a pandas DataFrame from transform and inverse transform
    (otherwise it will be a numpy array).
handle_missing: str
    how to handle missing values at fit time. Options are 'error', 'return_nan',
    and 'value'. Default 'value', which treat nans as a countable category at
    fit time.
handle_unknown: str, int or dict of {column : option, ...}.
    how to handle unknown labels at transform time. Options are 'error'
    'return_nan', 'value' and int. Defaults to None which uses nan behaviour
    specified at fit time. Passing an int will fill with this int value.
kwargs: dict.
    additional encoder specific parameters like regularisation.

