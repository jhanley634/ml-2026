#! /usr/bin/env python

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor

from solubility.eda import get_solubility_df

SEED = 42


def shuffle(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(frac=1, random_state=42)


pd.options.display.float_format = "{:.3f}".format


def create_model(*, want_charts: bool = False) -> None:
    text_cols = ["ID", "Name", "InChI", "InChIKey", "SMILES", "Group"]
    df = shuffle(get_solubility_df())
    df = df.drop(labels=text_cols, axis="columns")
    assert len(df) == 9_982, len(df)

    x = df.drop("Solubility", axis="columns")
    y = df["Solubility"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
    y_test = np.array(y_test, dtype=np.float64)

    model = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=SEED)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("RMSE:", mean_squared_error(y_test, y_pred))
    print("MAE: ", mean_absolute_error(y_test, y_pred))

    show_importance(model, x.columns)

    if want_charts:
        plot(y_test, y_pred)


def show_importance(
    model: XGBRegressor,
    feature_names: pd.Index,
) -> None:
    imp_series = pd.Series(
        data=model.feature_importances_,
        index=feature_names,
    )
    imp_df = imp_series.to_frame(name="Importance")
    imp_df = imp_df.sort_values(by="Importance", ascending=False)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print("\nTop Feature Importances:")
        print(imp_df.head(4))
    plot_tree(model)


TMP = Path("/tmp")


def _set_high_res_font_params() -> None:
    mpl.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 12,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
        },
    )


def plot_tree(
    model: XGBRegressor,
    out_file: Path = TMP / "tree.pdf",
    dpi: int = 1800,
) -> None:
    _set_high_res_font_params()
    xgb.plot_tree(model, tree_idx=0, rankdir="LR")
    plt.savefig(out_file, dpi=dpi, bbox_inches="tight")


def plot(y_test: NDArray[np.float64], y_pred: NDArray[np.float64]) -> None:

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, label="Predictions vs. Actual")
    plt.xlabel("Actual Solubility")
    plt.ylabel("Predicted Solubility")
    plt.title("XGBoost Solubility Prediction")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(y_test - y_pred, bins=20, kde=True)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Residual Plot")
    plt.show()


if __name__ == "__main__":
    create_model()
