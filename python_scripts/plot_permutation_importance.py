# %% [markdown]
#
# # Permutation Importance vs Random Forest Feature Importance (MDI)

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_context("poster")

# %% [markdown]
#
# ## Data Loading and Feature Engineering

# %%
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
rng = np.random.RandomState(seed=42)
X["random_cat"] = rng.randint(3, size=X.shape[0])
X["random_num"] = rng.randn(X.shape[0])

categorical_columns = ["pclass", "sex", "embarked", "random_cat"]
numerical_columns = ["age", "sibsp", "parch", "fare", "random_num"]

X = X[categorical_columns + numerical_columns]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

categorical_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1,
)
numerical_pipe = SimpleImputer(strategy="mean")

preprocessing = ColumnTransformer(
    [
        ("cat", categorical_encoder, categorical_columns),
        ("num", numerical_pipe, numerical_columns),
    ],
    verbose_feature_names_out=False,
)

rf = Pipeline(
    [
        ("preprocess", preprocessing),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)
rf.fit(X_train, y_train)

# %% [markdown]
#
# ## Accuracy of the Model

# %%
print(f"RF train accuracy: {rf.score(X_train, y_train):.3f}")
print(f"RF test accuracy: {rf.score(X_test, y_test):.3f}")

# %% [markdown]
#
# ## Tree's Feature Importance from Mean Decrease in Impurity (MDI)

# %%
import pandas as pd

feature_names = categorical_columns + numerical_columns

mdi_importance = pd.Series(
    rf[-1].feature_importances_, index=feature_names
).sort_values(ascending=True)

# %%
mdi_importance.plot.barh(figsize=(8, 6))
plt.title("Random Forest Feature Importances (MDI)")
plt.xlabel("Mean decrease in impurity")
plt.tight_layout()

# %% [markdown]
# ## Permutation Importance as alternative

# %%
from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)

sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=X.columns[sorted_importances_idx],
)
importances.plot.box(vert=False, whis=10, figsize=(8, 6))
plt.title("Permutation Importances (test set)")
plt.axvline(x=0, color="k", linestyle="--")
plt.xlabel("Decrease in accuracy score")
plt.tight_layout()

# %%
result = permutation_importance(
    rf, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
)

sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=X.columns[sorted_importances_idx],
)
importances.plot.box(vert=False, whis=10, figsize=(8, 6))
plt.title("Permutation Importances (train set)")
plt.axvline(x=0, color="k", linestyle="--")
plt.xlabel("Decrease in accuracy score")
plt.tight_layout()

# %% [markdown]
#
# ## What happen with non-overfitted forest?

# %%
rf.set_params(classifier__min_samples_leaf=20).fit(X_train, y_train)

# %%
print(f"RF train accuracy: {rf.score(X_train, y_train):.3f}")
print(f"RF test accuracy: {rf.score(X_test, y_test):.3f}")

# %%
importances = {}
for name, data, target in zip(["train", "test"], [X_train, X_test], [y_train, y_test]):
    result = permutation_importance(
        rf, data, target, n_repeats=10, random_state=42, n_jobs=2
    )
    if name == "train":
        sorted_importances_idx = result.importances_mean.argsort()

    importances[name] = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=data.columns[sorted_importances_idx],
    )
importances = pd.concat(importances, names=["set", "permutation"])

# %%
for name, data in importances.reset_index(level="set").groupby("set"):
    ax = data.plot.box(vert=False, whis=10, figsize=(8, 6))
    ax.set_title(f"Permutation Importances ({name} set)")
    ax.set_xlabel("Decrease in accuracy score")
    ax.axvline(x=0, color="k", linestyle="--")
    plt.tight_layout()
