import pandas as pd

def load_kepler(path: str = "Dataset\cumulative_2025.09.27_01.08.12.csv") -> pd.DataFrame:
    """Load NASA Kepler cumulative CSV robustly."""
    df = pd.read_csv(
        path,
        comment="#",
        sep=",",
        engine="python"
    )
    df.columns = df.columns.str.strip()
    return df
