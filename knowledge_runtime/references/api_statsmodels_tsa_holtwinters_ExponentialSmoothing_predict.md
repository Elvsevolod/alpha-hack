# statsmodels.tsa.holtwinters.ExponentialSmoothing.predict
Version: 0.14.0

## statsmodels.tsa.holtwinters.ExponentialSmoothing.predict(self, params, start=None, end=None)

In-sample and out-of-sample prediction.

Parameters
----------
params : ndarray
    The fitted model parameters.
start : int, str, or datetime
    Zero-indexed observation number at which to start forecasting, ie.,
    the first forecast is start. Can also be a date string to
    parse or a datetime type.
end : int, str, or datetime
    Zero-indexed observation number at which to end forecasting, ie.,
    the first forecast is start. Can also be a date string to
    parse or a datetime type.

Returns
-------
ndarray
    The predicted values.
