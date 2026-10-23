# category_encoders.TargetEncoder.fit_transform
Version: 2.6.4

## category_encoders.TargetEncoder.fit_transform(self, X, y=None, **fit_params)

Encoders that utilize the target must make sure that the training data are transformed with:
     transform(X, y)
and not with:
    transform(X)
