# missing_values
Source: https://raw.githubusercontent.com/catboost/catboost/v1.2.8/catboost/docs/en/concepts/algorithm-missing-values-processing.md

# Missing values processing

The missing values processing mode depends on the feature type and the selected package.


## Numerical features {#numerical-features}

CatBoost interprets the value of a numerical feature as a missing value if it is equal to one of the following values, which are package-dependant:

{% list tabs %}

- Python package

    - `None`
    - [Floating point NaN value](https://en.wikipedia.org/wiki/NaN)
    - One of the following strings when loading the values from files or as Python strings:

        
, <q>#N/A</q>, <q>#N/A N/A</q>, <q>#NA</q>, <q>-1.#IND</q>, <q>-1.#QNAN</q>, <q>-NaN</q>, <q>-nan</q>, <q>1.#IND</q>, <q>1.#QNAN</q>, <q>N/A</q>, <q>NA</q>, <q>NULL</q>, <q>NaN</q>, <q>n/a</q>, <q>nan</q>, <q>null</q>, <q>NAN</q>, <q>Na</q>, <q>na</q>, <q>Null</q>, <q>none</q>, <q>None</q>, <q>-</q>

This is an extended version of the default missing values list in [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).


- R package

    - [Floating point NaN value](https://en.wikipedia.org/wiki/NaN)
    - One of the following strings when loading the values from files:

        
, <q>#N/A</q>, <q>#N/A N/A</q>, <q>#NA</q>, <q>-1.#IND</q>, <q>-1.#QNAN</q>, <q>-NaN</q>, <q>-nan</q>, <q>1.#IND</q>, <q>1.#QNAN</q>, <q>N/A</q>, <q>NA</q>, <q>NULL</q>, <q>NaN</q>, <q>n/a</q>, <q>nan</q>, <q>null</q>, <q>NAN</q>, <q>Na</q>, <q>na</q>, <q>Null</q>, <q>none</q>, <q>None</q>, <q>-</q>

This is an extended version of the default missing values list in [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).


- Command-line version

  One of the following strings when loading the values from files when reading from an input file:

  
, <q>#N/A</q>, <q>#N/A N/A</q>, <q>#NA</q>, <q>-1.#IND</q>, <q>-1.#QNAN</q>, <q>-NaN</q>, <q>-nan</q>, <q>1.#IND</q>, <q>1.#QNAN</q>, <q>N/A</q>, <q>NA</q>, <q>NULL</q>, <q>NaN</q>, <q>n/a</q>, <q>nan</q>, <q>null</q>, <q>NAN</q>, <q>Na</q>, <q>na</q>, <q>Null</q>, <q>none</q>, <q>None</q>, <q>-</q>

This is an extended version of the default missing values list in [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).



{% endlist %}

The following modes for processing missing values are supported:


- "Forbidden" — Missing values are not supported, their presence is interpreted as an error.
- "Min" — Missing values are processed as the minimum value (less than all other values) for the feature. It is guaranteed that a split that separates missing values from all other values is considered when selecting trees.
- "Max" — Missing values are processed as the maximum value (greater than all other values) for the feature. It is guaranteed that a split that separates missing values from all other values is considered when selecting trees.




The default processing mode is Min. The methods for changing the default mode are package-dependant:

{% list tabs %}

- Python package

    - Globally for all features in the `nan_mode` [training parameter](../references/training-parameters/index.md).
    - Individually for each feature in the [Custom quantization borders and missing value modes](../concepts/input-data_custom-borders.md) input file. Such values override the global default setting.

-  R package

    - Globally for all features in the `nan_mode` [training parameter](../references/training-parameters/index.md).
    - Individually for each feature in the [Custom quantization borders and missing value modes](../concepts/input-data_custom-borders.md) input file. Such values override the global default setting.

- Command-line version

    - Globally for all features in the `--nan-mode` [training parameter](../references/training-parameters/index.md).
    - Individually for each feature in the [Custom quantization borders and missing value modes](../concepts/input-data_custom-borders.md) input file. Such values override the global default setting.

{% endlist %}

## Categorical features {#categorical-features}

CatBoost does not process categorical features in any specific way.
