# %% [markdown]
#
# # Exploring the SHAP library
#
# ## Loading the dataset and defining the predictive model

# %%
import seaborn as sns
sns.set_context("poster")

# %%
from sklearn.datasets import fetch_openml

survey = fetch_openml(data_id=534, as_frame=True)
survey.frame.head()

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    survey.data, survey.target, random_state=0
)

# %%
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector, make_column_transformer

categorical_columns = make_column_selector(dtype_include="category")
numerical_columns = make_column_selector(dtype_exclude="category")
preprocessor = make_column_transformer(
    (
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ),
        categorical_columns,
    ),
    remainder="passthrough",
    verbose_feature_names_out=False,
)
# %%
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

model = make_pipeline(
    preprocessor,
    HistGradientBoostingRegressor(max_iter=10_000, early_stopping=True, random_state=0),
)
model.fit(X_train, y_train)

# %%
from sklearn.metrics import mean_absolute_error

print(
    f"MAE on the training set: "
    f"{mean_absolute_error(y_train, model.predict(X_train)):.3f} $/hour"
)
print(
    f"MAE on the training set: "
    f"{mean_absolute_error(y_test, model.predict(X_test)):.3f} $/hour"
)

# %% [markdown]
#
# ## What SHAP values mean?

# %%
import pandas as pd

feature_names = categorical_columns(X_train) + numerical_columns(X_train)
X_train_preprocessed = pd.DataFrame(
    preprocessor.fit_transform(X_train), columns=feature_names
)
X_test_preprocessed = pd.DataFrame(
    preprocessor.transform(X_test), columns=feature_names
)

# %%
import shap

explainer = shap.Explainer(
    model[-1], masker=X_train_preprocessed, feature_perturbation="interventional"
)
shap_values = explainer(X_test_preprocessed)
shap_values

# %% [markdown]
#
# ## Explain model's prediction vs. mean prediction

# %%
model.predict(X_test.iloc[[0]])

# %% [markdown]
#
# The reported SHAP values for the different features are:

# %%
pd.Series(shap_values[0].values, index=feature_names)

# %% [markdown]
#
# Taking into account the base value, then the model prediction corresponds to
# the following sum:

# %%
shap_values[0].values.sum() + shap_values.base_values[0]

# %% [markdown]
#
# ## SHAP as a visual debugging tool
#
# SHAP package comes with handy plotting facilities to visualize the Shapley
# values.

# %%
shap.plots.waterfall(shap_values[0])

# %%
shap.initjs()
shap.plots.force(
    shap_values.base_values[0],
    shap_values.values[0],
    feature_names=feature_names,
)

# %%
shap.plots.beeswarm(shap_values)

# %% [markdown]
#
# ## Global explanation by averaging local explanations

# %%
shap.plots.bar(shap_values)

# %%
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

importances = permutation_importance(model, X_test, y_test, n_jobs=-1)
sorted_idx = importances.importances_mean.argsort()

importances = pd.DataFrame(
    importances.importances[sorted_idx].T, columns=X_test.columns[sorted_idx]
)
importances.plot.box(vert=False, whis=100, figsize=(8, 6))
plt.axvline(0, color="k", linestyle="--")
plt.xlabel("Decrease in R2 score")
_ = plt.title("Permutation importances")

# %% [markdown]
#
# We can make use of bootstrap resampling of the test set in order to repeat
# the experiment with a variation of the test dataset.

# %%
import numpy as np

rng = np.random.default_rng(42)
n_bootstrap = 25

all_shap_values = []
for _ in range(n_bootstrap):
    bootstrap_idx = rng.choice(
        np.arange(X_test.shape[0]), size=X_test.shape[0], replace=True
    )
    X_test_bootstrap = X_test.iloc[bootstrap_idx]
    X_test_preprocessed = pd.DataFrame(
        preprocessor.transform(X_test_bootstrap), columns=feature_names
    )
    all_shap_values.append(explainer(X_test_preprocessed))

# %%
shap_values = pd.DataFrame(
    [np.abs(shap_values.values).mean(axis=0) for shap_values in all_shap_values],
    columns=feature_names,
)
sorted_idx = shap_values.mean().sort_values().index

# %%
shap_values[sorted_idx].plot.box(vert=False, whis=10, figsize=(8, 6))
plt.xlabel("mean(|SHAP values|)")
_ = plt.title("SHAP values")

# %% [markdown]
#
# ## Be aware of some pitfalls regarding SHAP
#
# Pitfalls can come from:
#
# - some issues with the library
# - some limitations due to theoretical assumptions

# %%
explainer

# %%
explainer.feature_perturbation

# %%
explainer = shap.Explainer(model[-1])
explainer(X_test_preprocessed)

# %%
explainer.feature_perturbation

# %%
explainer = shap.Explainer(model[-1], feature_perturbation="interventional")
explainer(X_test_preprocessed)

# %%
explainer.feature_perturbation

# %%
X = np.concatenate([
    [[0, 0]] * 400,
    [[0, 1]] * 100,
    [[1, 0]] * 100,
    [[1, 1]] * 400
], axis=0)
X

# %%
y = np.array(
    [0] * 400 + [50] * 100 + [50] * 100 + [100] * 400
)

# %%
from sklearn.tree import DecisionTreeRegressor

tree_1 = DecisionTreeRegressor(random_state=0).fit(X, y)

# %%
from sklearn.tree import plot_tree

plt.figure(figsize=(10, 6))
_ = plot_tree(tree_1)

# %%
tree_2 = DecisionTreeRegressor(random_state=4).fit(X, y)

# %%
from sklearn.tree import plot_tree

plt.figure(figsize=(10, 6))
_ = plot_tree(tree_2)

# %%
X_test = np.array([[1, 1]])
explainer = shap.explainers.Exact(tree_1.predict, X)
explainer(X_test)

# %%
explainer = shap.explainers.Exact(
    tree_1.predict, masker=shap.maskers.Independent(X, max_samples=X.shape[0])
)
explainer(X_test)

# %%
explainer = shap.explainers.Exact(
    tree_2.predict, masker=shap.maskers.Independent(X, max_samples=X.shape[0])
)
explainer(X_test)

# %%
explainer = shap.Explainer(tree_1)
explainer(X_test)

# %%
explainer = shap.Explainer(tree_2)
explainer(X_test)

# %%
explainer

# %%
explainer = shap.Explainer(tree_1, shap.maskers.Independent(X, max_samples=X.shape[0]))
explainer(X_test)

# %%
explainer = shap.Explainer(tree_2, shap.maskers.Independent(X, max_samples=X.shape[0]))
explainer(X_test)
