# %% [markdown]
#
# # A couple of words regarding statistical modelling
#
# ## Statistical models
#
# ![book_model](../images/book_model.jpg)
#
# In general, we use statistical models as a simplification of the real
# underlying problem. We will first contrast different applications of
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
# For this specific model, we are not obsessed by the veracity of the forecasts
# but we want instead to understand the dynamics of the model and the sensitivity
# to specific parameters. We can therefore
# use the model to predict different scenarios, highlighting the different
# trends without attempting to predict perfectly the future.
#
# ### Predictive modelling
#
# When developing a predictive model, we really intend to have the most
# accurate predictions.

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
# ### Why explaining predictive model decisions?
#
# - Debug the model (e.g. assumption dataset, feature
#   engineering, hyperparameters, etc.) to make it work better.
# - Identify dependence on sensitive features that could reveal
#   robustness or fairness problems.
# - (⚠️) Potentially, draw conclusions about the real world assuming
#   the model is accurate enough.
#
# ### Association vs. causation
#
# Predictive models always find a statistical association (dependence)
# between `X` and `y`. Most machine-learning models do not attempt to
# quantify causal effects of `X` on `y`. In particular, most
# machine-learning models assume i.i.d. data and cannot guarantee to
# make accurate predictions under intervention.
#
# Machine-learning models can be used as building blocks to build
# causal models but this often requires additional assumptions on the
# underlying causal graph structure between the variables in `X` and `y`.
#
#
# Here are some specific frameworks intending to use machine-learning for
# causal inference:
#
# - https://github.com/microsoft/EconML
# - https://github.com/uber/causalml
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
# Courtesy to Ferenc Huszár:
# [video](https://www.youtube.com/watch?v=HOgx_SBBzn0&t=3855s&ab_channel=MLSSAfrica)
#
# ### A good explanation of a bad predictive model
#
# Before inspecting a model, you should always quantify its predictive power.

# %%
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

adult = fetch_openml("adult", version=2)
X, y = adult.data.select_dtypes(include="number"), adult.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(),
).fit(X_train, y_train)

# %%
import pandas as pd

coef = pd.Series(model[-1].coef_[0], index=X.columns)
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
cv_results[["train_score", "test_score"]].aggregate(["mean", "std"])

# %% [markdown]
#
# Explanations of bad models have no guarantee to give us valid
# information regarding the real world.
#
# ## Taxonomy of model explanation techniques
#
# ### Model specific vs. model agnostic
#
# Some models can be considered "glass-box" models: it is possible to directly
# access the association found between
# `X` and `y`. We can then inspect directly the model and this inspection
# is therefore model specific. For instance, a linear model exposes the
# coefficients of the regression.

# %%
coef = pd.Series(model[-1].coef_[0], index=X.columns)
_ = coef.plot.barh(figsize=(8, 6))

# %% [markdown]
#
# However, some techniques can be applied post-hoc to any type of models and
# it is therefore model agnostic. In particular, they are useful to gain
# insights from "black-box" models.

# %%
from sklearn.inspection import permutation_importance

importances = permutation_importance(
    model, X, y, scoring="balanced_accuracy", n_repeats=10, n_jobs=-1
)
importances = pd.DataFrame(importances.importances.T, columns=X.columns)

# %%
ax = importances.plot.box(vert=False, whis=10)
_ = ax.set_xlabel("Decrease in balanced accuracy")

# %% [markdown]
#
# ### Global explanation vs. local explanation
#
# The granularity of the explanation will also depend of the method used. Some
# methods are only computing a global explanation.

# %%
ax = importances.plot.box(vert=False, whis=10)
_ = ax.set_xlabel("Decrease in balanced accuracy")

# %% [markdown]
#
# Some others are computing a local explanation. This is then possible to get
# a more global explanation by averaging the local explanations.

# %%
import shap

X_train_preprocessed = pd.DataFrame(
    model[:-1].transform(X_train),
    columns=model[:-1].get_feature_names_out(),
)
X_test_preprocessed = pd.DataFrame(
    model[:-1].transform(X_test),
    columns=model[:-1].get_feature_names_out(),
)

explainer = shap.Explainer(model[-1], masker=X_train_preprocessed)
shap_values = explainer(X_test_preprocessed)

# %%
# explain first test data point
shap.plots.waterfall(shap_values[0])

# %%
# explain second test data point
shap.plots.waterfall(shap_values[1])

# %%
# global explanation by averaging local explanations
shap.plots.bar(shap_values)

ax = importances[importances.mean().sort_values().index].plot.box(vert=False, whis=10)
_ = ax.set_xlabel("Decrease in balanced accuracy")

# %% [markdown]
#
# ### Decision function explanation vs. loss explanation
#
# Some models will only explain the decision function of the model: at no
# point in time, the true variable `y` will be used to produce the explanation.
# Some other approaches will quantify the impact of input features on the ability
# to predict accurately the true `y`.
#
# ![shap_vs_sage](../images/shap_vs_sage.png)
#
# ## Overview of some explanation methods
#
# | Method name                               |  Model Agnostic |  Local explanation  |  Global explanation  | Decision function  |  Loss  |
# |-------------------------------------------|:----------------------:|:-------------------:|:--------------------:|:------------------:|:------:|
# | Linear model coefficients                 |           ❌           |         ✅          |         ✅           |         ✅          |   ❌   |
# | Tree-based mean decrease in impurity (MDI)|           ❌           |         ❌          |         ✅           |         ✅          |   ❌   |
# | Individual conditional expectation (ICE)  |           ✅           |         ✅          |         ❌           |         ✅          |   ❌   |
# | Partial dependence plot (PDP)             |           ✅           |         ❌          |         ✅           |         ✅          |   ❌   |
# | Permutation importance                    |           ✅           |         ❌          |         ✅           |         ❌          |   ✅   |
# | Shapley additive explanations (SHAP)      |           ✅           |         ✅          |         ✅           |         ✅          |   (✅) |
# | Shapley additive global importance (SAGE) |           ✅           |         ❌          |         ✅           |         ❌          |   ✅   |
# | Conterfactual explanations                |           ✅           |         ✅          |         ❌           |         ❌          |   ✅   |
#
# ### References
#
# [1] Pasqualino, Roberto, et al. "Understanding global systems today—A
# calibration of the World3-03 model between 1995 and 2012." Sustainability 7.8
# (2015): 9864-9889.
#
# [2] Molnar, Christoph. Interpretable machine learning. Lulu. com, 2020.
#
# [3] Covert, Ian, Scott M. Lundberg, and Su-In Lee. "Understanding global
# feature contributions with additive importance measures." Advances in Neural
# Information Processing Systems 33 (2020): 17212-17223.
