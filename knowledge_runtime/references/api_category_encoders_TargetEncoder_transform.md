# category_encoders.TargetEncoder.transform
Version: 2.6.4

## category_encoders.TargetEncoder.transform(self, X, y=None, override_return_df=False)

Perform the transformation to new categorical data.

Some encoders behave differently on whether y is given or not. This is mainly due to regularisation
in order to avoid overfitting.
On training data transform should be called with y, on test data without.

Parameters
----------

X : array-like, shape = [n_samples, n_features]
y : array-like, shape = [n_samples] or None
override_return_df : bool
    override self.return_df to force to return a data frame

Returns
-------

p : array or DataFrame, shape = [n_samples, n_features_out]
    Transformed values with encoding applied.
