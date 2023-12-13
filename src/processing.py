import polars as pl


def preprocessing(df: pl.DataFrame) -> pl.DataFrame:
    # デフォルトの"us"だとprophetの学習がなぜかうまくいかない
    df = df.with_columns(pl.col("Datetime").str.to_datetime(time_unit="ns")).rename(
        {"Datetime": "ds", "PJME_MW": "y"}
    )
    return df
