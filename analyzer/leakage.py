import pandas as pd

def check_leakage(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    correlations = data.corr(numeric_only=True)[target_column].drop(target_column)
    return correlations.abs().sort_values(ascending=False).reset_index().rename(columns={
        'index': 'Feature',
        target_column: 'Correlation with Target'
    })