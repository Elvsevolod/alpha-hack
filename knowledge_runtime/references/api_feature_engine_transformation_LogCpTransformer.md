# feature_engine.transformation.LogCpTransformer

Version: 1.8.3

feature_engine.transformation.LogCpTransformer(variables: Union[NoneType, int, str, List[Union[str, int]]] = None, base: str = 'e', C: Union[int, float, str, Dict[Union[str, int], Union[float, int]]] = 'auto') -> None

LogCpTransformer() applies the transformation log(x + C), where x is the
variable to transform and C is a positive constant. It can apply the natural
logarithm or the base 10 logarithm, where the natural logarithm is logarithm in
base e.

As the logarithm can only be applied to numerical non-negative values,
LogCpTransformer() extends the functionality of LogTransformer, by adding a
constant to shift the distribution of the variables towards positive values.

Note that if the variable contains a zero or a negative value after adding a
constant C, the transformer will return an error. This can occur if the values of
the variables in the test set are smaller than those seen during `fit()`.

A list of variables can be passed as an argument. Alternatively, the transformer
will automatically select and transform all variables of type numeric.

More details in the :ref:`User Guide <log_cp>`.

Parameters
----------
variables: list, default=None
    The list of numerical variables to transform. If None, the transformer
    will find and select all numerical variables. If C is a dictionary, then this
    parameter is ignored and the variables to transform are selected from the
    dictionary keys.

base: string, default='e'
    Indicates if the natural or base 10 logarithm should be applied. Can take
    values 'e' or '10'.

C: "auto", int or dict, default="auto"
    The constant C to add to the variable before the logarithm, i.e., log(x + C).

    - If int, then log(x + C)
    - If "auto", then C = abs(min(x)) + 1
    - If dict, dictionary mapping the constant C to apply to each variable.

    Note, when C is a dictionary, the parameter `variables` is ignored.

Attributes
----------
variables_:
    The group of variables that will be transformed.

C_:
    The constant C to add to each variable. If C = "auto" a dictionary with
    C = abs(min(variable)) + 1. For strictly positive variables, C = 0.

feature_names_in_:
    List with the names of features seen during `fit`.

n_features_in_:
    The number of features in the train set used in fit.

Methods
-------
fit:
    Learn the constant C.

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
    Transform the variables with the logarithm of x plus C.

Examples
--------

>>> import pandas as pd
>>> from feature_engine.transformation import LogCpTransformer
>>> X = pd.DataFrame(dict(
>>>    vara=[0, 1, 2, 3],
>>>    varb=[5, 5, 6, 7],
>>>    varc=[-2, -1, 0, 4],
>>>    vard=[-3, -2, -1, -5],
>>>    vare=["a", "b", "c", "d"]))
>>> lct = LogCpTransformer()
>>> lct.fit(X)
>>> X = lct.transform(X)
>>> X
       vara      varb      varc      vard vare
0  0.000000  1.609438  0.000000  1.098612    a
1  0.693147  1.609438  0.693147  1.386294    b
2  1.098612  1.791759  1.098612  1.609438    c
3  1.386294  1.945910  1.945910  0.000000    d

