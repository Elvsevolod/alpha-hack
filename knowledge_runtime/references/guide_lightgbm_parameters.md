# parameters
Source: https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html



  

  
  Parameters — LightGBM 4.6.0 documentation

  

      

      

    

    

 

    

      

          

      

      
Parameters

             View page source

  

          

             

Parameters

This page contains descriptions of all parameters in LightGBM.

List of other helpful links

Python API

Parameters Tuning

Parameters Format

Parameters are merged together in the following order (later items overwrite earlier ones):

LightGBM’s default values

special files for weight, init_score, query, and positions (see Others)

(CLI only) configuration in a file passed like config=train.conf

(CLI only) configuration passed via the command line

(Python, R) special keyword arguments to some functions (e.g. num_boost_round in train())

(Python, R) params function argument (including **kwargs in Python and ... in R)

(C API) parameters or params function argument

Many parameters have “aliases”, alternative names which refer to the same configuration.

Where a mix of the primary parameter name and aliases are given, the primary parameter name is always preferred to any aliases.

For example, in Python:

# use learning rate of 0.07, because 'learning_rate'
# is the primary parameter name
lgb.train(
   params={
      "learning_rate": 0.07,
      "shrinkage_rate": 0.12
   },
   train_set=dtrain
)

Where multiple aliases are given, and the primary parameter name is not, the first alias
appearing in the lists returned by Config::parameter2aliases() in the C++ library is used.
Those lists are hard-coded in a fairly arbitrary way… wherever possible, avoid relying on this behavior.

For example, in Python:

# use learning rate of 0.12, LightGBM has a hard-coded preference for 'shrinkage_rate'
# over any other aliases, and 'learning_rate' is not provided
lgb.train(
   params={
      "eta": 0.19,
      "shrinkage_rate": 0.12
   },
   train_set=dtrain
)

CLI

The parameters format is key1=value1 key2=value2 ....
Parameters can be set both in config file and command line.
By using command line, parameters should not have spaces before and after =.
By using config files, one line can only contain one parameter. You can use # to comment.

Python

Any parameters that accept multiple values should be passed as a Python list.

params = {
   "monotone_constraints": [-1, 0, 1]
}

R

Any parameters that accept multiple values should be passed as an R list.

params <- list(
   monotone_constraints = c(-1, 0, 1)
)

Core Parameters

config 🔗︎, default = "", type = string, aliases: config_file

path of config file

Note: can be used only in CLI version

task 🔗︎, default = train, type = enum, options: train, predict, convert_model, refit, aliases: task_type

train, for training, aliases: training

predict, for prediction, aliases: prediction, test

convert_model, for converting model file into if-else format, see more information in Convert Parameters

refit, for refitting existing models with new data, aliases: refit_tree

save_binary, load train (and validation) data then save dataset to binary file. Typical usage: save_binary first, then run multiple train tasks in parallel using the saved binary file

Note: can be used only in CLI version; for language-specific packages you can use the correspondent functions

objective 🔗︎, default = regression, type = enum, options: regression, regression_l1, huber, fair, poisson, quantile, mape, gamma, tweedie, binary, multiclass, multiclassova, cross_entropy, cross_entropy_lambda, lambdarank, rank_xendcg, aliases: objective_type, app, application, loss

regression application

regression, L2 loss, aliases: regression_l2, l2, mean_squared_error, mse, l2_root, root_mean_squared_error, rmse

regression_l1, L1 loss, aliases: l1, mean_absolute_error, mae

huber, Huber loss

fair, Fair loss

poisson, Poisson regression

quantile, Quantile regression

mape, MAPE loss, aliases: mean_absolute_percentage_error

gamma, Gamma regression with log-link. It might be useful, e.g., for modeling insurance claims severity, or for any target that might be gamma-distributed

tweedie, Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any target that might be tweedie-distributed

binary classification application

binary, binary log loss classification (or logistic regression)

requires labels in {0, 1}; see cross-entropy application for general probability labels in [0, 1]

multi-class classification application

multiclass, softmax objective function, aliases: softmax

multiclassova, One-vs-All binary objective function, aliases: multiclass_ova, ova, ovr

num_class should be set as well

cross-entropy application

cross_entropy, objective function for cross-entropy (with optional linear weights), aliases: xentropy

cross_entropy_lambda, alternative parameterization of cross-entropy, aliases: xentlambda

label is anything in interval [0, 1]

ranking application

lambdarank, lambdarank objective. label_gain can be used to set the gain (weight) of int label and all values in label must be smaller than number of elements in label_gain

rank_xendcg, XE_NDCG_MART ranking objective function, aliases: xendcg, xe_ndcg, xe_ndcg_mart, xendcg_mart

rank_xendcg is faster than and achieves the similar performance as lambdarank

label should be int type, and larger number represents the higher relevance (e.g. 0:bad, 1:fair, 2:good, 3:perfect)

custom objective function (gradients and hessians not computed directly by LightGBM)

custom

must be passed through parameters explicitly in the C API

Note: cannot be used in CLI version

boosting 🔗︎, default = gbdt, type = enum, options: gbdt, rf, dart, aliases: boosting_type, boost

gbdt, traditional Gradient Boosting Decision Tree, aliases: gbrt

rf, Random Forest, aliases: random_forest

dart, Dropouts meet Multiple Additive Regression Trees

Note: internally, LightGBM uses gbdt mode for the first 1 / learning_rate iterations

data_sample_strategy 🔗︎, default = bagging, type = enum, options: bagging, goss

bagging, Randomly Bagging Sampling

Note: bagging is only effective when bagging_freq > 0 and bagging_fraction < 1.0

goss, Gradient-based One-Side Sampling

New in version 4.0.0

data 🔗︎, default = "", type = string, aliases: train, train_data, train_data_file, data_filename

path of training data, LightGBM will train from this data

Note: can be used only in CLI version

valid 🔗︎, default = "", type = string, aliases: test, valid_data, valid_data_file, test_data, test_data_file, valid_filenames

path(s) of validation/test data, LightGBM will output metrics for these data

support multiple validation data, separated by ,

Note: can be used only in CLI version

num_iterations 🔗︎, default = 100, type = int, aliases: num_iteration, n_iter, num_tree, num_trees, num_round, num_rounds, nrounds, num_boost_round, n_estimators, max_iter, constraints: num_iterations >= 0

number of boosting iterations

Note: internally, LightGBM constructs num_class * num_iterations trees for multi-class classification problems

learning_rate 🔗︎, default = 0.1, type = double, aliases: shrinkage_rate, eta, constraints: learning_rate > 0.0

shrinkage rate

in dart, it also affects on normalization weights of dropped trees

num_leaves 🔗︎, default = 31, type = int, aliases: num_leaf, max_leaves, max_leaf, max_leaf_nodes, constraints: 1 < num_leaves <= 131072

max number of leaves in one tree

tree_learner 🔗︎, default = serial, type = enum, options: serial, feature, data, voting, aliases: tree, tree_type, tree_learner_type

serial, single machine tree learner

feature, feature parallel tree learner, aliases: feature_parallel

data, data parallel tree learner, aliases: data_parallel

voting, voting parallel tree learner, aliases: voting_parallel

refer to Distributed Learning Guide to get more details

num_threads 🔗︎, default = 0, type = int, aliases: num_thread, nthread, nthreads, n_jobs

used only in train, prediction and refit tasks or in correspondent functions of language-specific packages

number of threads for LightGBM

0 means default number of threads in OpenMP

for the best speed, set this to the number of real CPU cores, not the number of threads (most CPUs use hyper-threading to generate 2 threads per CPU core)

do not set it too large if your dataset is small (for instance, do not use 64 threads for a dataset with 10,000 rows)

be aware a task manager or any similar CPU monitoring tool might report that cores not being fully utilized. This is normal

for distributed learning, do not use all CPU cores because this will cause poor performance for the network communication

Note: please don’t change this during training, especially when running multiple jobs simultaneously by external packages, otherwise it may cause undesirable errors

device_type 🔗︎, default = cpu, type = enum, options: cpu, gpu, cuda, aliases: device

device for the tree learning

cpu supports all LightGBM functionality and is portable across the widest range of operating systems and hardware

cuda offers faster training than gpu or cpu, but only works on GPUs supporting CUDA

gpu can be faster than cpu and works on a wider range of GPUs than CUDA

Note: it is recommended to use the smaller max_bin (e.g. 63) to get the better speed up

Note: for the faster speed, GPU uses 32-bit float point to sum up by default, so this may affect the accuracy for some tasks. You can set gpu_use_dp=true to enable 64-bit float point, but it will slow down the training

Note: refer to Installation Guide to build LightGBM with GPU or CUDA support

seed 🔗︎, default = None, type = int, aliases: random_seed, random_state

this seed is used to generate other seeds, e.g. data_random_seed, feature_fraction_seed, etc.

by default, this seed is unused in favor of default values of other seeds

this seed has lower priority in comparison with other seeds, which means that it will be overridden, if you set other seeds explicitly

deterministic 🔗︎, default = false, type = bool

used only with cpu device type

setting this to true should ensure the stable results when using the same data and the same parameters (and different num_threads)

when you use the different seeds, different LightGBM versions, the binaries compiled by different compilers, or in different systems, the results are expected to be different

you can raise issues in LightGBM GitHub repo when you meet the unstable results

Note: setting this to true may slow down the training

Note: to avoid potential instability due to numerical issues, please set force_col_wise=true or force_row_wise=true when setting deterministic=true

Learning Control Parameters

force_col_wise 🔗︎, default = false, type = bool

used only with cpu device type

set this to true to force col-wise histogram building

enabling this is recommended when:

the number of columns is large, or the total number of bins is large

num_threads is large, e.g. > 20

you want to reduce memory cost

Note: when both force_col_wise and force_row_wise are false, LightGBM will firstly try them both, and then use the faster one. To remove the overhead of testing set the faster one to true manually

Note: this parameter cannot be used at the same time with force_row_wise, choose only one of them

force_row_wise 🔗︎, default = false, type = bool

used only with cpu device type

set this to true to force row-wise histogram building

enabling this is recommended when:

the number of data points is large, and the total number of bins is relatively small

num_threads is relatively small, e.g. <= 16

you want to use small bagging_fraction or goss sample strategy to speed up

Note: setting this to true will double the memory cost for Dataset object. If you have not enough memory, you can try setting force_col_wise=true

Note: when both force_col_wise and force_row_wise are false, LightGBM will firstly try them both, and then use the faster one. To remove the overhead of testing set the faster one to true manually

Note: this parameter cannot be used at the same time with force_col_wise, choose only one of them

histogram_pool_size 🔗︎, default = -1.0, type = double, aliases: hist_pool_size

max cache size in MB for historical histogram

< 0 means no limit

max_depth 🔗︎, default = -1, type = int

limit the max depth for tree model. This is used to deal with over-fitting when #data is small. Tree still grows leaf-wise

<= 0 means no limit

min_data_in_leaf 🔗︎, default = 20, type = int, aliases: min_data_per_leaf, min_data, min_child_samples, min_samples_leaf, constraints: min_data_in_leaf >= 0

minimal number of data in one leaf. Can be used to deal with over-fitting

Note: this is an approximation based on the Hessian, so occasionally you may observe splits which produce leaf nodes that have less than this many observations

min_sum_hessian_in_leaf 🔗︎, default = 1e-3, type = double, aliases: min_sum_hessian_per_leaf, min_sum_hessian, min_hessian, min_child_weight, constraints: min_sum_hessian_in_leaf >= 0.0

minimal sum hessian in one leaf. Like min_data_in_leaf, it can be used to deal with over-fitting

bagging_fraction 🔗︎, default = 1.0, type = double, aliases: sub_row, subsample, bagging, constraints: 0.0 < bagging_fraction <= 1.0

like feature_fraction, but this will randomly select part of data without resampling

can be used to speed up training

can be used to deal with over-fitting

Note: to enable bagging, bagging_freq should be set to a non zero value as well

pos_bagging_fraction 🔗︎, default = 1.0, type = double, aliases: pos_sub_row, pos_subsample, pos_bagging, constraints: 0.0 < pos_bagging_fraction <= 1.0

used only in binary application

used for imbalanced binary classification problem, will randomly sample #pos_samples * pos_bagging_fraction positive samples in bagging

should be used together with neg_bagging_fraction

set this to 1.0 to disable

Note: to enable this, you need to set bagging_freq and neg_bagging_fraction as well

Note: if both pos_bagging_fraction and neg_bagging_fraction are set to 1.0,  balanced bagging is disabled

Note: if balanced bagging is enabled, bagging_fraction will be ignored

neg_bagging_fraction 🔗︎, default = 1.0, type = double, aliases: neg_sub_row, neg_subsample, neg_bagging, constraints: 0.0 < neg_bagging_fraction <= 1.0

used only in binary application

used for imbalanced binary classification problem, will randomly sample #neg_samples * neg_bagging_fraction negative samples in bagging

should be used together with pos_bagging_fraction

set this to 1.0 to disable

Note: to enable this, you need to set bagging_freq and pos_bagging_fraction as well

Note: if both pos_bagging_fraction and neg_bagging_fraction are set to 1.0,  balanced bagging is disabled

Note: if balanced bagging is enabled, bagging_fraction will be ignored

bagging_freq 🔗︎, default = 0, type = int, aliases: subsample_freq

frequency for bagging

0 means disable bagging; k means perform bagging at every k iteration. Every k-th iteration, LightGBM will randomly select bagging_fraction * 100% of the data to use for the next k iterations

Note: bagging is only effective when 0.0 < bagging_fraction < 1.0

bagging_seed 🔗︎, default = 3, type = int, aliases: bagging_fraction_seed

random seed for bagging

bagging_by_query 🔗︎, default = false, type = bool

whether to do bagging sample by query

New in version 4.6.0

feature_fraction 🔗︎, default = 1.0, type = double, aliases: sub_feature, colsample_bytree, constraints: 0.0 < feature_fraction <= 1.0

LightGBM will randomly select a subset of features on each iteration (tree) if feature_fraction is smaller than 1.0. For example, if you set it to 0.8, LightGBM will select 80% of features before training each tree

can be used to speed up training

can be used to deal with over-fitting

feature_fraction_bynode 🔗︎, default = 1.0, type = double, aliases: sub_feature_bynode, colsample_bynode, constraints: 0.0 < feature_fraction_bynode <= 1.0

LightGBM will randomly select a subset of features on each tree node if feature_fraction_bynode is smaller than 1.0. For example, if you set it to 0.8, LightGBM will select 80% of features at each tree node

can be used to deal with over-fitting

Note: unlike feature_fraction, this cannot speed up training

Note: if both feature_fraction and feature_fraction_bynode are smaller than 1.0, the final fraction of each node is feature_fraction * feature_fraction_bynode

feature_fraction_seed 🔗︎, default = 2, type = int

random seed for feature_fraction

extra_trees 🔗︎, default = false, type = bool, aliases: extra_tree

use extremely randomized trees

if set to true, when evaluating node splits LightGBM will check only one randomly-chosen threshold for each feature

can be used to speed up training

can be used to deal with over-fitting

extra_seed 🔗︎, default = 6, type = int

random seed for selecting thresholds when extra_trees is true

early_stopping_round 🔗︎, default = 0, type = int, aliases: early_stopping_rounds, early_stopping, n_iter_no_change

will stop training if one metric of one validation data doesn’t improve in last early_stopping_round rounds

<= 0 means disable

can be used to speed up training

early_stopping_min_delta 🔗︎, default = 0.0, type = double, constraints: early_stopping_min_delta >= 0.0

when early stopping is used (i.e. early_stopping_round > 0), require the early stopping metric to improve by at least this delta to be considered an improvement

New in version 4.4.0

first_metric_only 🔗︎, default = false, type = bool

LightGBM allows you to provide multiple evaluation metrics. Set this to true, if you want to use only the first metric for early stopping

max_delta_step 🔗︎, default = 0.0, type = double, aliases: max_tree_output, max_leaf_output

used to limit the max output of tree leaves

<= 0 means no constraint

the final max output of leaves is learning_rate * max_delta_step

lambda_l1 🔗︎, default = 0.0, type = double, aliases: reg_alpha, l1_regularization, constraints: lambda_l1 >= 0.0

L1 regularization

lambda_l2 🔗︎, default = 0.0, type = double, aliases: reg_lambda, lambda, l2_regularization, constraints: lambda_l2 >= 0.0

L2 regularization

linear_lambda 🔗︎, default = 0.0, type = double, constraints: linear_lambda >= 0.0

linear tree regularization, corresponds to the parameter lambda in Eq. 3 of Gradient Boosting with Piece-Wise Linear Regression Trees

min_gain_to_split 🔗︎, default = 0.0, type = double, aliases: min_split_gain, constraints: min_gain_to_split >= 0.0

the minimal gain to perform split

can be used to speed up training

drop_rate 🔗︎, default = 0.1, type = double, aliases: rate_drop, constraints: 0.0 <= drop_rate <= 1.0

used only in dart

dropout rate: a fraction of previous trees to drop during the dropout

max_drop 🔗︎, default = 50, type = int

used only in dart

max number of dropped trees during one boosting iteration

<=0 means no limit

skip_drop 🔗︎, default = 0.5, type = double, constraints: 0.0 <= skip_drop <= 1.0

used only in dart

probability of skipping the dropout procedure during a boosting iteration

xgboost_dart_mode 🔗︎, default = false, type = bool

used only in dart

set this to true, if you want to use XGBoost DART mode

uniform_drop 🔗︎, default = false, type = bool

used only in dart

set this to true, if you want to use uniform drop

drop_seed 🔗︎, default = 4, type = int

used only in dart

random seed to choose dropping models

top_rate 🔗︎, default = 0.2, type = double, constraints: 0.0 <= top_rate <= 1.0

used only in goss

the retain ratio of large gradient data

other_rate 🔗︎, default = 0.1, type = double, constraints: 0.0 <= other_rate <= 1.0

used only in goss

the retain ratio of small gradient data

min_data_per_group 🔗︎, default = 100, type = int, constraints: min_data_per_group > 0

used for the categorical features

minimal number of data per categorical group

max_cat_threshold 🔗︎, default = 32, type = int, constraints: max_cat_threshold > 0

used for the categorical features

limit number of split points considered for categorical features. See the documentation on how LightGBM finds optimal splits for categorical features for more details

can be used to speed up training

cat_l2 🔗︎, default = 10.0, type = double, constraints: cat_l2 >= 0.0

used for the categorical features

L2 regularization in categorical split

cat_smooth 🔗︎, default = 10.0, type = double, constraints: cat_smooth >= 0.0

used for the categorical features

this can reduce the effect of noises in categorical features, especially for categories with few data

max_cat_to_onehot 🔗︎, default = 4, type = int, constraints: max_cat_to_onehot > 0

used for the categorical features

when number of categories of one feature smaller than or equal to max_cat_to_onehot, one-vs-other split algorithm will be used

top_k 🔗︎, default = 20, type = int, aliases: topk, constraints: top_k > 0

used only in voting tree learner, refer to Voting parallel

set this to larger value for more accurate result, but it will slow down the training speed

monotone_constraints 🔗︎, default = None, type = multi-int, aliases: mc, monotone_constraint, monotonic_cst

used for constraints of monotonic features

1 means increasing, -1 means decreasing, 0 means non-constraint

you need to specify all features in order. For example, mc=-1,0,1 means decreasing for the 1st feature, non-constraint for the 2nd feature and increasing for the 3rd feature

monotone_constraints_method 🔗︎, default = basic, type = enum, options: basic, intermediate, advanced, aliases: monotone_constraining_method, mc_method

used only if monotone_constraints is set

monotone constraints method

basic, the most basic monotone constraints method. It does not slow down the training speed at all, but over-constrains the predictions

intermediate, a more advanced method, which may slow down the training speed very slightly. However, this method is much less constraining than the basic method and should significantly improve the results

advanced, an even more advanced method, which may slow down the training speed. However, this method is even less constraining than the intermediate method and should again significantly improve the results

monotone_penalty 🔗︎, default = 0.0, type = double, aliases: monotone_splits_penalty, ms_penalty, mc_penalty, constraints: monotone_penalty >= 0.0

used only if monotone_constraints is set

monotone penalty: a penalization parameter X forbids any monotone splits on the first X (rounded down) level(s) of the tree. The penalty applied to monotone splits on a given depth is a continuous, increasing function the penalization parameter

if 0.0 (the default), no penalization is applied

feature_contri 🔗︎, default = None, type = multi-double, aliases: feature_contrib, fc, fp, feature_penalty

used to control feature’s split gain, will use gain[i] = max(0, feature_contri[i]) * gain[i] to replace the split gain of i-th feature

you need to specify all features in order

forcedsplits_filename 🔗︎, default = "", type = string, aliases: fs, forced_splits_filename, forced_splits_file, forced_splits

path to a .json file that specifies splits to force at the top of every decision tree before best-first learning commences

.json file can be arbitrarily nested, and each split contains feature, threshold fields, as well as left and right fields representing subsplits

categorical splits are forced in a one-hot fashion, with left representing the split containing the feature value and right representing other values

Note: the forced split logic will be ignored, if the split makes gain worse

see this file as an example

refit_decay_rate 🔗︎, default = 0.9, type = double, constraints: 0.0 <= refit_decay_rate <= 1.0

decay rate of refit task, will use leaf_output = refit_decay_rate * old_leaf_output + (1.0 - refit_decay_rate) * new_leaf_output to refit trees

used only in refit task in CLI version or as argument in refit function in language-specific package

cegb_tradeoff 🔗︎, default = 1.0, type = double, constraints: cegb_tradeoff >= 0.0

cost-effective gradient boosting multiplier for all penalties

cegb_penalty_split 🔗︎, default = 0.0, type = double, constraints: cegb_penalty_split >= 0.0

cost-effective gradient-boosting penalty for splitting a node

cegb_penalty_feature_lazy 🔗︎, default = 0,0,...,0, type = multi-double

cost-effective gradient boosting penalty for using a feature

applied per data point

cegb_penalty_feature_coupled 🔗︎, default = 0,0,...,0, type = multi-double

cost-effective gradient boosting penalty for using a feature

applied once per forest

path_smooth 🔗︎, default = 0, type = double, constraints: path_smooth >=  0.0

controls smoothing applied to tree nodes

helps prevent overfitting on leaves with few samples

if 0.0 (the default), no smoothing is applied

if path_smooth > 0 then min_data_in_leaf must be at least 2

larger values give stronger regularization

the weight of each node is w * (n / path_smooth) / (n / path_smooth + 1) + w_p / (n / path_smooth + 1), where n is the number of samples in the node, w is the optimal node weight to minimise the loss (approximately -sum_gradients / sum_hessians), and w_p is the weight of the parent node

note that the parent output w_p itself has smoothing applied, unless it is the root node, so that the smoothing effect accumulates with the tree depth

interaction_constraints 🔗︎, default = "", type = string

controls which features can appear in the same branch

by default interaction constraints are disabled, to enable them you can specify

for CLI, lists separated by commas, e.g. [0,1,2],[2,3]

for Python-package, list of lists, e.g. [[0, 1, 2], [2, 3]]

for R-package, list of character or numeric vectors, e.g. list(c("var1", "var2", "var3"), c("var3", "var4")) or list(c(1L, 2L, 3L), c(3L, 4L)). Numeric vectors should use 1-based indexing, where 1L is the first feature, 2L is the second feature, etc.

any two features can only appear in the same branch only if there exists a constraint containing both features

verbosity 🔗︎, default = 1, type = int, aliases: verbose

controls the level of LightGBM’s verbosity

< 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug

input_model 🔗︎, default = "", type = string, aliases: model_input, model_in

filename of input model

for prediction task, this model will be applied to prediction data

for train task, training will be continued from this model

Note: can be used only in CLI version

output_model 🔗︎, default = LightGBM_model.txt, type = string, aliases: model_output, model_out

filename of output model in training

Note: can be used only in CLI version

saved_feature_importance_type 🔗︎, default = 0, type = int

the feature importance type in the saved model file

0: count-based feature importance (numbers of splits are counted); 1: gain-based feature importance (values of gain are counted)

Note: can be used only in CLI version

snapshot_freq 🔗︎, default = -1, type = int, aliases: save_period

frequency of saving model file snapshot

set this to positive value to enable this function. For example, the model file will be snapshotted at each iteration if snapshot_freq=1

Note: can be used only in CLI version

use_quantized_grad 🔗︎, default = false, type = bool

whether to use gradient quantization when training

enabling this will discretize (quantize) the gradients and hessians into bins of num_grad_quant_bins

with quantized training, most arithmetics in the training process will be integer operations

gradient quantization can accelerate training, with little accuracy drop in most cases

Note: works only with cpu and cuda device type

New in version 4.0.0

num_grad_quant_bins 🔗︎, default = 4, type = int

used only if use_quantized_grad=true

number of bins to quantization gradients and hessians

with more bins, the quantized training will be closer to full precision training

Note: works only with cpu and cuda device type

New in version 4.0.0

quant_train_renew_leaf 🔗︎, default = false, type = bool

used only if use_quantized_grad=true

whether to renew the leaf values with original gradients when quantized training

renewing is very helpful for good quantized training accuracy for ranking objectives

Note: works only with cpu and cuda device type

New in version 4.0.0

stochastic_rounding 🔗︎, default = true, type = bool

used only if use_quantized_grad=true

whether to use stochastic rounding in gradient quantization

Note: works only with cpu and cuda device type

New in version 4.0.0

IO Parameters

Dataset Parameters

linear_tree 🔗︎, default = false, type = bool, aliases: linear_trees

fit piecewise linear gradient boosting tree

tree splits are chosen in the usual way, but the model at each leaf is linear instead of constant

the linear model at each leaf includes all the numerical features in that leaf’s branch

the first tree has constant leaf values

categorical features are used for splits as normal but are not used in the linear models

missing values should not be encoded as 0. Use np.nan for Python, NA for the CLI, and NA, NA_real_, or NA_integer_ for R

it is recommended to rescale data before training so that features have similar mean and standard deviation

Note: works only with cpu, gpu device type and serial tree learner

Note: regression_l1 objective is not supported with linear tree boosting

Note: setting linear_tree=true significantly increases the memory use of LightGBM

Note: if you specify monotone_constraints, constraints will be enforced when choosing the split points, but not when fitting the linear models on leaves

max_bin 🔗︎, default = 255, type = int, aliases: max_bins, constraints: max_bin > 1

max number of bins that feature values will be bucketed in

small number of bins may reduce training accuracy but may increase general power (deal with over-fitting)

LightGBM will auto compress memory according to max_bin. For example, LightGBM will use uint8_t for feature value if max_bin=255

max_bin_by_feature 🔗︎, default = None, type = multi-int

max number of bins for each feature

if not specified, will use max_bin for all features

min_data_in_bin 🔗︎, default = 3, type = int, constraints: min_data_in_bin > 0

minimal number of data inside one bin

use this to avoid one-data-one-bin (potential over-fitting)

bin_construct_sample_cnt 🔗︎, default = 200000, type = int, aliases: subsample_for_bin, constraints: bin_construct_sample_cnt > 0

number of data that sampled to construct feature discrete bins

setting this to larger value will give better training result, but may increase data loading time

set this to larger value if data is very sparse

Note: don’t set this to small values, otherwise, you may encounter unexpected errors and poor accuracy

data_random_seed 🔗︎, default = 1, type = int, aliases: data_seed

random seed for sampling data to construct histogram bins

is_enable_sparse 🔗︎, default = true, type = bool, aliases: is_sparse, enable_sparse, sparse

used to enable/disable sparse optimization

enable_bundle 🔗︎, default = true, type = bool, aliases: is_enable_bundle, bundle

set this to false to disable Exclusive Feature Bundling (EFB), which is described in LightGBM: A Highly Efficient Gradient Boosting Decision Tree

Note: disabling this may cause the slow training speed for sparse datasets

use_missing 🔗︎, default = true, type = bool

set this to false to disable the special handle of missing value

zero_as_missing 🔗︎, default = false, type = bool

set this to true to treat all zero as missing values (including the unshown values in LibSVM / sparse matrices)

set this to false to use na for representing missing values

feature_pre_filter 🔗︎, default = true, type = bool

set this to true (the default) to tell LightGBM to ignore the features that are unsplittable based on min_data_in_leaf

as dataset object is initialized only once and cannot be changed after that, you may need to set this to false when searching parameters with min_data_in_leaf, otherwise features are filtered by min_data_in_leaf firstly if you don’t reconstruct dataset object

Note: setting this to false may slow down the training

pre_partition 🔗︎, default = false, type = bool, aliases: is_pre_partition

used for distributed learning (excluding the feature_parallel mode)

true if training data are pre-partitioned, and different machines use different partitions

two_round 🔗︎, default = false, type = bool, aliases: two_round_loading, use_two_round_loading

set this to true if data file is too big to fit in memory

by default, LightGBM will map data file to memory and load features from memory. This will provide faster data loading speed, but may cause run out of memory error when the data file is very big

Note: works only in case of loading data directly from text file

header 🔗︎, default = false, type = bool, aliases: has_header

set this to true if input data has header

Note: works only in case of loading data directly from text file

label_column 🔗︎, default = "", type = int or string, aliases: label

used to specify the label column

use number for index, e.g. label=0 means column_0 is the label

add a prefix name: for column name, e.g. label=name:is_click

if omitted, the first column in the training data is used as the label

Note: works only in case of loading data directly from text file

weight_column 🔗︎, default = "", type = int or string, aliases: weight

used to specify the weight column

use number for index, e.g. weight=0 means column_0 is the weight

add a prefix name: for column name, e.g. weight=name:weight

Note: works only in case of loading data directly from text file

Note: index starts from 0 and it doesn’t count the label column when passing type is int, e.g. when label is column_0, and weight is column_1, the correct parameter is weight=0

Note: weights should be non-negative

group_column 🔗︎, default = "", type = int or string, aliases: group, group_id, query_column, query, query_id

used to specify the query/group id column

use number for index, e.g. query=0 means column_0 is the query id

add a prefix name: for column name, e.g. query=name:query_id

Note: works only in case of loading data directly from text file

Note: data should be grouped by query_id, for more information, see Query Data

Note: index starts from 0 and it doesn’t count the label column when passing type is int, e.g. when label is column_0 and query_id is column_1, the correct parameter is query=0

ignore_column 🔗︎, default = "", type = multi-int or string, aliases: ignore_feature, blacklist

used to specify some ignoring columns in training

use number for index, e.g. ignore_column=0,1,2 means column_0, column_1 and column_2 will be ignored

add a prefix name: for column name, e.g. ignore_column=name:c1,c2,c3 means c1, c2 and c3 will be ignored

Note: works only in case of loading data directly from text file

Note: index starts from 0 and it doesn’t count the label column when passing type is int

Note: despite the fact that specified columns will be completely ignored during the training, they still should have a valid format allowing LightGBM to load file successfully

categorical_feature 🔗︎, default = "", type = multi-int or string, aliases: cat_feature, categorical_column, cat_column, categorical_features

used to specify categorical features

use number for index, e.g. categorical_feature=0,1,2 means column_0, column_1 and column_2 are categorical features

add a prefix name: for column name, e.g. categorical_feature=name:c1,c2,c3 means c1, c2 and c3 are categorical features

Note: all values will be cast to int32 (integer codes will be extracted from pandas categoricals in the Python-package)

Note: index starts from 0 and it doesn’t count the label column when passing type is int

Note: all values should be less than Int32.MaxValue (2147483647)

Note: using large values could be memory consuming. Tree decision rule works best when categorical features are presented by consecutive integers starting from zero

Note: all negative values will be treated as missing values

Note: the output cannot be monotonically constrained with respect to a categorical feature

Note: floating point numbers in categorical features will be rounded towards 0

forcedbins_filename 🔗︎, default = "", type = string

path to a .json file that specifies bin upper bounds for some or all features

.json file should contain an array of objects, each containing the word feature (integer feature index) and bin_upper_bound (array of thresholds for binning)

see this file as an example

save_binary 🔗︎, default = false, type = bool, aliases: is_save_binary, is_save_binary_file

if true, LightGBM will save the dataset (including validation data) to a binary file. This speed ups the data loading for the next time

Note: init_score is not saved in binary file

Note: can be used only in CLI version; for language-specific packages you can use the correspondent function

precise_float_parser 🔗︎, default = false, type = bool

use precise floating point number parsing for text parser (e.g. CSV, TSV, LibSVM input)

Note: setting this to true may lead to much slower text parsing

parser_config_file 🔗︎, default = "", type = string

path to a .json file that specifies customized parser initialized configuration

see lightgbm-transform for usage examples

Note: lightgbm-transform is not maintained by LightGBM’s maintainers. Bug reports or feature requests should go to issues page

New in version 4.0.0

Predict Parameters

start_iteration_predict 🔗︎, default = 0, type = int

used only in prediction task

used to specify from which iteration to start the prediction

<= 0 means from the first iteration

num_iteration_predict 🔗︎, default = -1, type = int

used only in prediction task

used to specify how many trained iterations will be used in prediction

<= 0 means no limit

predict_raw_score 🔗︎, default = false, type = bool, aliases: is_predict_raw_score, predict_rawscore, raw_score

used only in prediction task

set this to true to predict only the raw scores

set this to false to predict transformed scores

predict_leaf_index 🔗︎, default = false, type = bool, aliases: is_predict_leaf_index, leaf_index

used only in prediction task

set this to true to predict with leaf index of all trees

predict_contrib 🔗︎, default = false, type = bool, aliases: is_predict_contrib, contrib

used only in prediction task

set this to true to estimate SHAP values, which represent how each feature contributes to each prediction

produces #features + 1 values where the last value is the expected value of the model output over the training data

Note: if you want to get more explanation for your model’s predictions using SHAP values like SHAP interaction values, you can install shap package

Note: unlike the shap package, with predict_contrib we return a matrix with an extra column, where the last column is the expected value

Note: this feature is not implemented for linear trees

predict_disable_shape_check 🔗︎, default = false, type = bool

used only in prediction task

control whether or not LightGBM raises an error when you try to predict on data with a different number of features than the training data

if false (the default), a fatal error will be raised if the number of features in the dataset you predict on differs from the number seen during training

if true, LightGBM will attempt to predict on whatever data you provide. This is dangerous because you might get incorrect predictions, but you could use it in situations where it is difficult or expensive to generate some features and you are very confident that they were never chosen for splits in the model

Note: be very careful setting this parameter to true

pred_early_stop 🔗︎, default = false, type = bool

used only in prediction task

used only in classification and ranking applications

used only for predicting normal or raw scores

if true, will use early-stopping to speed up the prediction. May affect the accuracy

Note: cannot be used with rf boosting type or custom objective function

pred_early_stop_freq 🔗︎, default = 10, type = int

used only in prediction task and if pred_early_stop=true

the frequency of checking early-stopping prediction

pred_early_stop_margin 🔗︎, default = 10.0, type = double

used only in prediction task and if pred_early_stop=true

the threshold of margin in early-stopping prediction

output_result 🔗︎, default = LightGBM_predict_result.txt, type = string, aliases: predict_result, prediction_result, predict_name, prediction_name, pred_name, name_pred

used only in prediction task

filename of prediction result

Note: can be used only in CLI version

Convert Parameters

convert_model_language 🔗︎, default = "", type = string

used only in convert_model task

only cpp is supported yet; for conversion model to other languages consider using m2cgen utility

if convert_model_language is set and task=train, the model will be also converted

Note: can be used only in CLI version

convert_model 🔗︎, default = gbdt_prediction.cpp, type = string, aliases: convert_model_file

used only in convert_model task

output filename of converted model

Note: can be used only in CLI version

Objective Parameters

objective_seed 🔗︎, default = 5, type = int

used only in rank_xendcg objective

random seed for objectives, if random process is needed

num_class 🔗︎, default = 1, type = int, aliases: num_classes, constraints: num_class > 0

used only in multi-class classification application

is_unbalance 🔗︎, default = false, type = bool, aliases: unbalance, unbalanced_sets

used only in binary and multiclassova applications

set this to true if training data are unbalanced

Note: while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities

Note: this parameter cannot be used at the same time with scale_pos_weight, choose only one of them

scale_pos_weight 🔗︎, default = 1.0, type = double, constraints: scale_pos_weight > 0.0

used only in binary and multiclassova applications

weight of labels with positive class

Note: while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities

Note: this parameter cannot be used at the same time with is_unbalance, choose only one of them

sigmoid 🔗︎, default = 1.0, type = double, constraints: sigmoid > 0.0

used only in binary and multiclassova classification and in lambdarank applications

parameter for the sigmoid function

boost_from_average 🔗︎, default = true, type = bool

used only in regression, binary, multiclassova and cross-entropy applications

adjusts initial score to the mean of labels for faster convergence

reg_sqrt 🔗︎, default = false, type = bool

used only in regression application

used to fit sqrt(label) instead of original values and prediction result will be also automatically converted to prediction^2

might be useful in case of large-range labels

alpha 🔗︎, default = 0.9, type = double, constraints: alpha > 0.0

used only in huber and quantile regression applications

parameter for Huber loss and Quantile regression

fair_c 🔗︎, default = 1.0, type = double, constraints: fair_c > 0.0

used only in fair regression application

parameter for Fair loss

poisson_max_delta_step 🔗︎, default = 0.7, type = double, constraints: poisson_max_delta_step > 0.0

used only in poisson regression application

parameter for Poisson regression to safeguard optimization

tweedie_variance_power 🔗︎, default = 1.5, type = double, constraints: 1.0 <= tweedie_variance_power < 2.0

used only in tweedie regression application

used to control the variance of the tweedie distribution

set this closer to 2 to shift towards a Gamma distribution

set this closer to 1 to shift towards a Poisson distribution

lambdarank_truncation_level 🔗︎, default = 30, type = int, constraints: lambdarank_truncation_level > 0

used only in lambdarank application

controls the number of top-results to focus on during training, refer to “truncation level” in the Sec. 3 of LambdaMART paper

this parameter is closely related to the desirable cutoff k in the metric NDCG@k that we aim at optimizing the ranker for. The optimal setting for this parameter is likely to be slightly higher than k (e.g., k + 3) to include more pairs of documents to train on, but perhaps not too high to avoid deviating too much from the desired target metric NDCG@k

lambdarank_norm 🔗︎, default = true, type = bool

used only in lambdarank application

set this to true to normalize the lambdas for different queries, and improve the performance for unbalanced data

set this to false to enforce the original lambdarank algorithm

label_gain 🔗︎, default = 0,1,3,7,15,31,63,...,2^30-1, type = multi-double

used only in lambdarank application

relevant gain for labels. For example, the gain of label 2 is 3 in case of default label gains

separate by ,

lambdarank_position_bias_regularization 🔗︎, default = 0.0, type = double, constraints: lambdarank_position_bias_regularization >= 0.0

used only in lambdarank application when positional information is provided and position bias is modeled

larger values reduce the inferred position bias factors

New in version 4.1.0

Metric Parameters

metric 🔗︎, default = "", type = multi-enum, aliases: metrics, metric_types

metric(s) to be evaluated on the evaluation set(s)

"" (empty string or not specified) means that metric corresponding to specified objective will be used (this is possible only for pre-defined objective functions, otherwise no evaluation metric will be added)

"None" (string, not a None value) means that no metric will be registered, aliases: na, null, custom

l1, absolute loss, aliases: mean_absolute_error, mae, regression_l1

l2, square loss, aliases: mean_squared_error, mse, regression_l2, regression

rmse, root square loss, aliases: root_mean_squared_error, l2_root

quantile, Quantile regression

mape, MAPE loss, aliases: mean_absolute_percentage_error

huber, Huber loss

fair, Fair loss

poisson, negative log-likelihood for Poisson regression

gamma, negative log-likelihood for Gamma regression

gamma_deviance, residual deviance for Gamma regression

tweedie, negative log-likelihood for Tweedie regression

ndcg, NDCG, aliases: lambdarank, rank_xendcg, xendcg, xe_ndcg, xe_ndcg_mart, xendcg_mart

map, MAP, aliases: mean_average_precision

auc, AUC

average_precision, average precision score

binary_logloss, log loss, aliases: binary

binary_error, for one sample: 0 for correct classification, 1 for error classification

auc_mu, AUC-mu

multi_logloss, log loss for multi-class classification, aliases: multiclass, softmax, multiclassova, multiclass_ova, ova, ovr

multi_error, error rate for multi-class classification

cross_entropy, cross-entropy (with optional linear weights), aliases: xentropy

cross_entropy_lambda, “intensity-weighted” cross-entropy, aliases: xentlambda

kullback_leibler, Kullback-Leibler divergence, aliases: kldiv

support multiple metrics, separated by ,

metric_freq 🔗︎, default = 1, type = int, aliases: output_freq, constraints: metric_freq > 0

frequency for metric output

Note: can be used only in CLI version

is_provide_training_metric 🔗︎, default = false, type = bool, aliases: training_metric, is_training_metric, train_metric

set this to true to output metric result over training dataset

Note: can be used only in CLI version

eval_at 🔗︎, default = 1,2,3,4,5, type = multi-int, aliases: ndcg_eval_at, ndcg_at, map_eval_at, map_at

used only with ndcg and map metrics

NDCG and MAP evaluation positions, separated by ,

multi_error_top_k 🔗︎, default = 1, type = int, constraints: multi_error_top_k > 0

used only with multi_error metric

threshold for top-k multi-error metric

the error on each sample is 0 if the true class is among the top multi_error_top_k predictions, and 1 otherwise

more precisely, the error on a sample is 0 if there are at least num_classes - multi_error_top_k predictions strictly less than the prediction on the true class

when multi_error_top_k=1 this is equivalent to the usual multi-error metric

auc_mu_weights 🔗︎, default = None, type = multi-double

used only with auc_mu metric

list representing flattened matrix (in row-major order) giving loss weights for classification errors

list should have n * n elements, where n is the number of classes

the matrix co-ordinate [i, j] should correspond to the i * n + j-th element of the list

if not specified, will use equal weights for all classes

Network Parameters

num_machines 🔗︎, default = 1, type = int, aliases: num_machine, constraints: num_machines > 0

the number of machines for distributed learning application

this parameter is needed to be set in both socket and MPI versions

local_listen_port 🔗︎, default = 12400 (random for Dask-package), type = int, aliases: local_port, port, constraints: local_listen_port > 0

TCP listen port for local machines

Note: don’t forget to allow this port in firewall settings before training

time_out 🔗︎, default = 120, type = int, constraints: time_out > 0

socket time-out in minutes

machine_list_filename 🔗︎, default = "", type = string, aliases: machine_list_file, machine_list, mlist

path of file that lists machines for this distributed learning application

each line contains one IP and one port for one machine. The format is ip port (space as a separator)

Note: can be used only in CLI version

machines 🔗︎, default = "", type = string, aliases: workers, nodes

list of machines in the following format: ip1:port1,ip2:port2

GPU Parameters

gpu_platform_id 🔗︎, default = -1, type = int

used only with gpu device type

OpenCL platform ID. Usually each GPU vendor exposes one OpenCL platform

-1 means the system-wide default platform

Note: refer to GPU Targets for more details

gpu_device_id 🔗︎, default = -1, type = int

OpenCL device ID in the specified platform or CUDA device ID. Each GPU in the selected platform has a unique device ID

-1 means the default device in the selected platform

Note: refer to GPU Targets for more details

gpu_use_dp 🔗︎, default = false, type = bool

set this to true to use double precision math on GPU (by default single precision is used)

Note: can be used only in OpenCL implementation (device_type="gpu"), in CUDA implementation only double precision is currently supported

num_gpu 🔗︎, default = 1, type = int, constraints: num_gpu > 0

number of GPUs

Note: can be used only in CUDA implementation (device_type="cuda")

Others

Continued Training with Input Score

LightGBM supports continued training with initial scores.
It uses an additional file to store these initial scores, like the following:

0.5
-0.1
0.9
...

It means the initial score of the first data row is 0.5, second is -0.1, and so on.
The initial score file corresponds with data file line by line, and has per score per line.

If the name of data file is train.txt, the initial score file should be named as train.txt.init and placed in the same folder as the data file.
In this case, LightGBM will auto load initial score file if it exists.

If binary data files exist for raw data file train.txt, for example in the name train.txt.bin, then the initial score file should be named as train.txt.bin.init.

Weight Data

LightGBM supports weighted training.
It uses an additional file to store weight data, like the following:

1.0
0.5
0.8
...

It means the weight of the first data row is 1.0, second is 0.5, and so on. Weights should be non-negative.

The weight file corresponds with data file line by line, and has per weight per line.

And if the name of data file is train.txt, the weight file should be named as train.txt.weight and placed in the same folder as the data file.
In this case, LightGBM will load the weight file automatically if it exists.

Also, you can include weight column in your data file.
Please refer to the weight_column parameter in above.

Query Data

For learning to rank, it needs query information for training data.

LightGBM uses an additional file to store query data, like the following:

27
18
67
...

For wrapper libraries like in Python and R, this information can also be provided as an array-like via the Dataset parameter group.

[27, 18, 67, ...]

For example, if you have a 112-document dataset with group = [27, 18, 67], that means that you have 3 groups, where the first 27 records are in the first group, records 28-45 are in the second group, and records 46-112 are in the third group.

Note: data should be ordered by the query.

If the name of data file is train.txt, the query file should be named as train.txt.query and placed in the same folder as the data file.
In this case, LightGBM will load the query file automatically if it exists.

Also, you can include query/group id column in your data file.
Please refer to the group_column parameter in above.

           

          

      

  

