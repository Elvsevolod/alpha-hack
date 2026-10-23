# flaml.AutoML.predict_proba
Version: 2.3.6

## flaml.AutoML.predict_proba(self, X, **pred_kwargs)

Predict the probability of each class from features, only works for
classification problems.

Args:
    X: A numpy array of featurized instances, shape n * m.
    **pred_kwargs: Other key word arguments to pass to predict_proba() function of
        the searched learners, such as per_device_eval_batch_size.

Returns:
    A numpy array of shape n * c. c is the  # classes. Each element at
    (i, j) is the probability for instance i to be in class j.
