# %% [markdown]
#
# # A couple of words regarding statistical modelling
#
# ## Statistical models
#
# ![book_model](../images/book_model.jpg)
#
# In general, we use statistical models as a simplification of the real
# underlying problem. We will first contrast to different applications of
# statistical modelling: inference vs. prediction.
#
# ### Statistical inference
#
# Can we understand the dynamics of a given modelisation of the world.

# %%
import seaborn as sns

sns.set_context("poster")

# %%
from pyworld3 import World3

world3 = World3(pyear=2022)
world3.init_world3_constants()
world3.init_world3_variables()
world3.set_world3_table_functions()
world3.set_world3_delay_functions()
world3.run_world3()

# %%
import matplotlib.pyplot as plt
from pyworld3.utils import plot_world_variables

plot_world_variables(
    world3.time,
    [world3.nrfr, world3.iopc, world3.fpc, world3.pop, world3.ppolx],
    ["NRFR", "IOPC", "FPC", "POP", "PPOLX"],
    [[0, 1], [0, 1e3], [0, 1e3], [0, 16e9], [0, 32]],
    figsize=(16, 10),
    title="World3 standard run",
)
axes = plt.gcf().get_axes()
handles = [ax.get_lines()[0] for ax in axes]
labels = [
    "Resource",
    "Industrial output\n per capita",
    "Food per capita",
    "Population",
    "Pollution",
]
_ = plt.legend(handles, labels, loc="upper right")

# %% [markdown]
#
# In inference, we are not generally obsessed by the veracity of the results
# but we want instead understand the dynamics of the model. We can therefore
# used the model to predict different scenarios, highlighting the different
# trends but never at predicting perfectly the future.
#
# ### Predictive modelling
#
# When developing a predictive model, we really intend to have the most
# accurate model at prediting future values.

# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = load_iris(as_frame=True)
X, y = iris.data[["sepal width (cm)", "petal width (cm)"]], iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LogisticRegression().fit(X_train, y_train)

# %%
from sklearn.inspection import DecisionBoundaryDisplay

_, ax = plt.subplots(figsize=(10, 8))
display = DecisionBoundaryDisplay.from_estimator(model, X_train, alpha=0.5, ax=ax)
scatter = display.ax_.scatter(
    X_train["sepal width (cm)"], X_train["petal width (cm)"], c=y_train, edgecolor="k"
)
display.ax_.legend(scatter.legend_elements()[0], iris.target_names)
_ = display.ax_.set_title("Prediction on training data")

# %%
_, ax = plt.subplots(figsize=(10, 8))
display = DecisionBoundaryDisplay.from_estimator(model, X_test, alpha=0.5, ax=ax)
scatter = display.ax_.scatter(
    X_test["sepal width (cm)"], X_test["petal width (cm)"], c=y_test, edgecolor="k"
)
display.ax_.legend(scatter.legend_elements()[0], iris.target_names)
_ = display.ax_.set_title("Prediction on test data")

# %% [markdown]
#
# We aim at finding the best predictive model, the one that generalizes best on
# unseen future data.
#
# ### Association vs. causation
#
# Predictive models always find an association between `X` and `y`. They never
# find a causation: they use probability distributions and do not use causal
# graphs.
#
# Let's consider 3 scenarios that would produce the same `X` and `y`
# distributions:

# %%
import numpy as np


def generate_data(scenario, n_samples=1_000, seed=None):
    rng = np.random.default_rng(seed)
    if scenario == "X causes y":
        X = rng.normal(size=(n_samples,))
        y = X + 1 + np.sqrt(3) * rng.normal(size=(n_samples,))
    elif scenario == "y causes X":
        y = 1 + 2 * rng.normal(size=(n_samples,))
        X = (y - 1) / 4 + np.sqrt(3) * rng.normal(size=(n_samples,)) / 2
    elif scenario == "Z causes X and y":
        Z = rng.normal(size=(n_samples,))
        y = Z + 1 + np.sqrt(3) * rng.normal(size=(n_samples,))
        X = Z
    return X, y


# %%
for scenario, color in zip(
    ["X causes y", "y causes X", "Z causes X and y"],
    ["tab:blue", "tab:orange", "tab:green"],
):
    X, y = generate_data(scenario=scenario, n_samples=100, seed=0)
    joint_plot = sns.jointplot(x=X, y=y, color=color)
    joint_plot.set_axis_labels(xlabel=scenario)

# %% [markdown]
#
# In terms of distributions, the 3 scenarios are identical. But what causes
# `X` and `y` are different. Applying a simple linear model on these different
# dataset will provide the following results:


# %%
for scenario, color in zip(
    ["X causes y", "y causes X", "Z causes X and y"],
    ["tab:blue", "tab:orange", "tab:green"],
):
    X, y = generate_data(scenario=scenario, n_samples=100, seed=0)
    joint_plot = sns.jointplot(x=X, y=y, color=color, kind="reg")
    joint_plot.set_axis_labels(xlabel=scenario)

# %% [markdown]
#
# ### A predictive model can always be inspected
#
# Before inspecting a model, you should always quantify its predictive power.

# %%
from sklearn.datasets import fetch_openml

adult = fetch_openml("adult", version=2)
X, y = adult.data.select_dtypes(include="number"), adult.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LogisticRegression().fit(X_train, y_train)

# %%
import pandas as pd

coef = pd.Series(model.coef_[0], index=X.columns)
_ = coef.plot.barh(figsize=(8, 6))

# %%
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
cv_results = pd.DataFrame(
    cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring="balanced_accuracy",
        return_train_score=True,
        n_jobs=-1,
    )
)

# %%
cv_results[["train_score", "test_score"]].aggregate(["mean", "std"])

# %% [markdown]
#
# ## Taxonomy of model explanation techniques
#
# ### Model specific vs. model agnostic
#
# Some models carry some information regarding the association found between
# `X` and `y`. We can then inspect directly the model and this inspection
# is therefore model specific. For instance, a linear model exposes the
# coefficients of the regression.

# %%
coef = pd.Series(model.coef_[0], index=X.columns)
_ = coef.plot.barh(figsize=(8, 6))

# %% [markdown]
#
# However, some techniques can be applied post-hoc to any type of models and
# it is therefore model agnostic.

# %%
from sklearn.inspection import permutation_importance

importances = permutation_importance(
    model, X, y, scoring="balanced_accuracy", n_repeats=10, n_jobs=-1
)
importances = pd.DataFrame(importances.importances.T, columns=X.columns)

# %%
ax = importances.plot.box(vert=False, whis=10)
ax.set_xlabel("Decrease in balanced accuracy")

# %% [markdown]
#
# ### Global explanation vs. local explanation
#
# The granularity of the explanation will also depend of the method used. Some
# methods are only computing a global explanation.

# %%
ax = importances.plot.box(vert=False, whis=10)
ax.set_xlabel("Decrease in balanced accuracy")

# %% [markdown]
#
# Some others are computing a local explanation. This is then possible to get
# a more global explanation by averaging the local explanations.

# %%
import shap

explainer = shap.Explainer(model, masker=X_train)
shap_values = explainer(X_test)

# %%
shap.plots.waterfall(shap_values[0])

# %% [markdown]
#
# ### Decision function explanation vs. loss explanation
#
# ![shap_vs_sage](../images/shap_vs_sage.png)
#
# Some models will only explain the decision function of the model: at no
# point in time, the true variable `y` will be used to produce the explanation.
# Some other approaches will use the true `y` to compute the explanation.
#
# ## Summary
#
# | Method name                               |  Agnostic vs. Specific |  Local explanation  |  Global explanation  | Decision function  |  Loss  |
# |-------------------------------------------|:----------------------:|:-------------------:|:--------------------:|:------------------:|:------:|
# | Linear model coefficients                 |           ❌           |         ✅          |         ✅           |         ✅          |   ❌   |
# | Mean decrease in impurity (MDI)           |           ❌           |         ❌          |         ✅           |         ✅          |   ❌   |
# | Individual conditional expectation (ICE)  |           ✅           |         ✅          |         ❌           |         ✅          |   ❌   |
# | Partial dependence plot (PDP)             |           ✅           |         ❌          |         ✅           |         ✅          |   ❌   |
# | Permutation importance                    |           ✅           |         ❌          |         ✅           |         ❌          |   ✅   |
# | Shapley additive explanations (SHAP)      |           ✅           |         ✅          |         ✅           |         ✅          |   ✅!  |
# | Shapley additive global importance (SAGE) |           ✅           |         ❌          |         ✅           |         ❌          |   ✅   |
