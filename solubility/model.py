#! /usr/bin/env python

import pandas as pd

from solubility.eda import get_solubility_df


def shuffle(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(frac=1, random_state=42)


def model() -> None:
    text_cols = ["Name", "InChI", "InChIKey", "SMILES"]
    df = shuffle(get_solubility_df())
    df = df.drop(labels=text_cols, axis="columns")

    assert len(df) == 9_982, len(df)
    test_idx = int(0.8 * len(df))
    train = df[:test_idx]
    test = df[test_idx:]

    print(train[["Group", "Solubility"]].describe())
    print(
        train.drop(
            labels=[
                "NumHAcceptors",
                "NumHDonors",
                "NumHeteroatoms",
                "NumRotatableBonds",
                "NumValenceElectrons",
                "NumAromaticRings",
                "NumSaturatedRings",
                "NumAliphaticRings",
                "RingCount",
                "Ocurrences",
                "TPSA",
                "LabuteASA",
                "BalabanJ",
                "BertzCT",
            ],
            axis="columns",
        )
    )


if __name__ == "__main__":
    model()
