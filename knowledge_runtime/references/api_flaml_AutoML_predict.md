# flaml.AutoML.predict
Version: 2.3.6

## flaml.AutoML.predict(self, X: 'np.array | DataFrame | list[str] | list[list[str]] | psDataFrame', **pred_kwargs)

Predict label from features.

Args:
    X: A numpy array or pandas dataframe or pyspark.pandas dataframe
    of featurized instances, shape n * m,
        or for time series forcast tasks:
            a pandas dataframe with the first column containing
            timestamp values (datetime type) or an integer n for
            the predict steps (only valid when the estimator is
            arima or sarimax). Other columns in the dataframe
            are assumed to be exogenous variables (categorical
            or numeric).
    **pred_kwargs: Other key word arguments to pass to predict() function of
        the searched learners, such as per_device_eval_batch_size.

```python
multivariate_X_test = DataFrame({
    'timeStamp': pd.date_range(start='1/1/2022', end='1/07/2022'),
    'categorical_col': ['yes', 'yes', 'no', 'no', 'yes', 'no', 'yes'],
    'continuous_col': [105, 107, 120, 118, 110, 112, 115]
})
model.predict(multivariate_X_test)
```

Returns:
    A array-like of shape n * 1: each element is a predicted
    label for an instance.
