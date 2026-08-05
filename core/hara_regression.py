"""Nutrient content (N/P/K/Mg) regression for the eHara feature.

Adapted from the original standalone ``haraLR.py`` calibration script.
Given a training Excel file (historical ground-truth leaf nutrient
measurements paired with band reflectance values) and a set of new points
(already extracted from a raster), this module fits a PCA + Linear
Regression model per nutrient and predicts calibrated values for the new
points.

By design this re-trains from scratch on every run — mirroring the
original script's behaviour, the "model" is really a per-dataset
calibration rather than a persisted/pretrained network, since the
training data (and therefore the calibration) may change from one field
survey to the next.

Formulas (same convention as the original script, where band1/band2/band3
are reflectance values and band3 is the NIR-like band):
    NDVI  = (band3 - band1) / (band3 + band1)
    GNDVI = (band3 - band2) / (band3 + band2)
    SR    = band3 / band1
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


class HaraRegressionError(Exception):
    """Raised when training data or prediction input is invalid."""


# Feature columns used to build the polynomial-expanded regression input.
# Order matters (must match between training and prediction), so we always
# select columns by this explicit list rather than relying on whatever
# order columns happen to appear in an Excel file.
INDEPENDENT_VARS = ["GNDVI", "band1", "band2", "band3", "NDVI", "SR"]

# Target nutrient columns expected in the training data.
NUTRIENTS = ["N", "P", "K", "Mg"]

# Plausible value ranges (leaf %) used to rescale/clip raw regression
# output. Taken directly from the original haraLR.py calibration
# (adjust_values_n / _p / _k / _mg).
NUTRIENT_RANGES = {
    "N": (2.25, 3.25),
    "P": (0.10, 0.30),
    "K": (0.65, 1.25),
    "Mg": (0.15, 0.80),
}

REQUIRED_TRAINING_COLUMNS = NUTRIENTS + INDEPENDENT_VARS

# Polynomial degrees 1..5, matching applyingLRModelWithReadExcel in the
# original script (range(1, 6)).
POLY_DEGREE_RANGE = range(1, 6)


def calculate_ndvi(band1: pd.Series, band3: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (band3 - band1) / (band3 + band1)


def calculate_gndvi(band2: pd.Series, band3: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (band3 - band2) / (band3 + band2)


def calculate_sr(band1: pd.Series, band3: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        return band3 / band1


def _add_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with var_degree_N columns added for every
    independent variable/degree combination (original columns are kept)."""
    df = df.copy()
    for var in INDEPENDENT_VARS:
        for degree in POLY_DEGREE_RANGE:
            df[f"{var}_degree_{degree}"] = df[var] ** degree
    return df


def _adjust(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Rescale raw linear-regression output (assumed roughly in [0, 1])
    into the biologically plausible [low, high] range and clip outliers."""
    scaled = values * (high - low) + low
    return np.clip(scaled, low, high)


def load_training_data(path: str) -> pd.DataFrame:
    """Load and validate the training Excel file.

    Expected columns: ID, X, Y, N, P, K, Mg, band1, band2, band3, NDVI,
    GNDVI, SR (ID/X/Y are optional/unused for fitting).
    """
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    missing = [c for c in REQUIRED_TRAINING_COLUMNS if c not in df.columns]
    if missing:
        raise HaraRegressionError(
            "Training data is missing required column(s): " + ", ".join(missing)
        )

    # Coerce feature/target columns to numeric; anything unparsable becomes
    # NaN which we then drop so a stray text cell doesn't crash PCA/fit.
    numeric_cols = REQUIRED_TRAINING_COLUMNS
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    before = len(df)
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    dropped = before - len(df)

    if df.empty:
        raise HaraRegressionError(
            "Training data has no valid rows after removing rows with "
            "missing/non-numeric values in required columns."
        )
    if dropped:
        # Not fatal, but the caller may want to log/inform the user.
        df.attrs["rows_dropped"] = dropped

    return df


class HaraRegressionModel:
    """Fits PCA + one LinearRegression per nutrient on the given training
    data, then predicts all 4 nutrients for new points."""

    def __init__(self, training_df: pd.DataFrame):
        if len(training_df) < 2:
            raise HaraRegressionError(
                "Training data must contain at least 2 valid rows to fit a model."
            )

        feature_df = _add_polynomial_features(training_df[INDEPENDENT_VARS].astype(float))

        self.pca = PCA()
        x_pca = self.pca.fit_transform(feature_df)

        self.models: dict[str, LinearRegression] = {}
        for nutrient in NUTRIENTS:
            model = LinearRegression()
            model.fit(x_pca, training_df[nutrient])
            self.models[nutrient] = model

    def predict(self, points_df: pd.DataFrame) -> pd.DataFrame:
        """``points_df`` must contain band1, band2, band3, NDVI, GNDVI, SR
        columns. Returns a DataFrame (same index as points_df) with
        'N Leaf (%)', 'P Leaf (%)', 'K Leaf (%)', 'Mg Leaf (%)' columns."""
        missing = [c for c in INDEPENDENT_VARS if c not in points_df.columns]
        if missing:
            raise HaraRegressionError(
                "Points to predict are missing required column(s): " + ", ".join(missing)
            )

        feature_df = _add_polynomial_features(points_df[INDEPENDENT_VARS].astype(float))
        x_pca = self.pca.transform(feature_df)

        result = pd.DataFrame(index=points_df.index)
        for nutrient in NUTRIENTS:
            raw = self.models[nutrient].predict(x_pca)
            low, high = NUTRIENT_RANGES[nutrient]
            adjusted = _adjust(raw, low, high)
            result[f"{nutrient} Leaf (%)"] = np.round(adjusted, 2)
        return result
