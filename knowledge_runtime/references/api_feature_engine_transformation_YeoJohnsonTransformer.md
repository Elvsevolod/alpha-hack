# feature_engine.transformation.YeoJohnsonTransformer

Version: 1.8.3

feature_engine.transformation.YeoJohnsonTransformer(variables: Union[NoneType, int, str, List[Union[str, int]]] = None) -> None

The YeoJohnsonTransformer() applies the Yeo-Johnson transformation to the
numerical variables.

The Yeo-Johnson transformation implemented by this transformer is that of
SciPy.stats:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html

The YeoJohnsonTransformer() works only with numerical variables.

A list of variables can be passed as an argument. Alternatively, the
transformer will automatically select and transform all numerical
variables.

More details in the :ref:`User Guide <yeojohnson>`.

Parameters
----------
variables: list, default=None
    The list of numerical variables to transform. If None, the transformer will
    automatically find and select all numerical variables.

Attributes
----------
lambda_dict_
    Dictionary containing the best lambda for the Yeo-Johnson per variable.

variables_:
    The group of variables that will be transformed.

feature_names_in_:
    List with the names of features seen during `fit`.

n_features_in_:
    The number of features in the train set used in fit.

Methods
-------
fit:
    Learn the optimal lambda for the Yeo-Johnson transformation.

fit_transform:
    Fit to data, then transform it.

get_feature_names_out:
    Get output feature names for transformation.

get_params:
    Get parameters for this estimator.

set_params:
    Set the parameters of this estimator.

inverse_transform:
    Convert the data back to the original representation.

transform:
    Apply the Yeo-Johnson transformation.

References
----------
.. [1] Yeo, In-Kwon and Johnson, Richard (2000).
    A new family of power transformations to improve normality or symmetry.
    Biometrika, 87, 954-959.

.. [2] Weisberg S. "Yeo-Johnson Power Transformations".
    https://www.stat.umn.edu/arc/yjpower.pdf

Examples
--------

>>> import numpy as np
>>> import pandas as pd
>>> from feature_engine.transformation import YeoJohnsonTransformer
>>> np.random.seed(42)
>>> X = pd.DataFrame(dict(x = np.random.lognormal(size = 100) - 10))
>>> yjt = YeoJohnsonTransformer()
>>> yjt.fit(X)
>>> X = yjt.transform(X)
>>> X.head()
               x
0 -267042.906453
1 -444357.138990
2 -221626.115742
3  -23647.632651
4 -467264.993249

