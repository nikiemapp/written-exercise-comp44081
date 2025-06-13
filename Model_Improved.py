import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import statsmodels.api as sm


class RedemptionSalesImprovedModel:
    def __init__(self, X: pd.DataFrame, target_col: str):
        self.X = X.sort_index()
        self.target_col = target_col
        self.results = {}
        self.preds = {}
        self.models = {}
        self.feature_importances_ = {}

    @staticmethod
    def _metrics(y_true, y_pred):
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return {
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE": mape,
            "R2": r2_score(y_true, y_pred),
        }

    def _log(self, model, split, y_true, y_pred, fitted=None):
        self.results.setdefault(model, {})[split] = self._metrics(y_true, y_pred)
        self.preds.setdefault(model, {})[split] = y_pred
        if fitted is not None:
            self.models.setdefault(model, {})[split] = fitted

    def _base(self, y_train, y_test, split):
        stl = sm.tsa.seasonal_decompose(y_train, period=365)
        seas = pd.Series(stl.seasonal, index=y_train.index).clip(lower=0)
        doy_means = seas.groupby(seas.index.dayofyear).mean()
        y_hat = pd.Series(doy_means.reindex(y_test.index.dayofyear).values, index=y_test.index)
        self._log("Base", split, y_test, y_hat)

    def _rf(self, X_train, X_test, y_train, y_test, split):
        model = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_hat = pd.Series(model.predict(X_test), index=y_test.index)
        self._log("RandomForest", split, y_test, y_hat, model)
        self.feature_importances_["RandomForest"] = model.feature_importances_

    def _lgbm(self, X_train, X_test, y_train, y_test, split):
        model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, subsample=0.6, max_depth=70,
                                  num_leaves=64, random_state=42)
        model.fit(X_train, y_train)
        y_hat = pd.Series(model.predict(X_test), index=y_test.index)
        self._log("LightGBM", split, y_test, y_hat, model)
        self.feature_importances_["LightGBM"] = model.feature_importances_

    def _xgb(self, X_train, X_test, y_train, y_test, split):
        model = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                                 colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_hat = pd.Series(model.predict(X_test), index=y_test.index)
        self._log("XGBoost", split, y_test, y_hat, model)
        self.feature_importances_["XGBoost"] = model.feature_importances_

    def _ensemble(self, y_test, split):
        rf = self.preds["RandomForest"][split]
        lg = self.preds["LightGBM"][split]
        xb = self.preds["XGBoost"][split]
        ens = (rf + lg + xb) / 3
        self._log("Ensemble", split, y_test, ens)

    def run(self, n_splits=4, test_size=365):
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        for split, (train_idx, test_idx) in enumerate(tscv.split(self.X)):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = X_train[self.target_col], X_test[self.target_col]
            feats_train = X_train.drop(columns=[self.target_col])
            feats_test = X_test.drop(columns=[self.target_col])

            self._base(y_train, y_test, split)
            self._rf(feats_train, feats_test, y_train, y_test, split)
            self._lgbm(feats_train, feats_test, y_train, y_test, split)
            self._xgb(feats_train, feats_test, y_train, y_test, split)
            self._ensemble(y_test, split)

        self.performance_results = (
            pd.DataFrame({m: pd.DataFrame(d).T.mean() for m, d in self.results.items()})
            .T.sort_values("MAPE")
        )

        return self

    def metrics(self):
        return self.performance_results

    def plot(self):
        for model in ["Base", "RandomForest", "LightGBM", "XGBoost", "Ensemble"]:
            for split in self.preds.get(model, {}):
                preds = self.preds[model][split]
                plt.figure(figsize=(15, 4))
                plt.plot(self.X[self.target_col], '.', ms=2, color="grey", label="Observed")
                plt.plot(preds, lw=2, color="red", label=f"{model} Split {split}")
                plt.title(f"{model} - CV split {split}")
                plt.legend(); plt.tight_layout(); plt.show()

    def importance(self):
        feature_names = self.X.drop(columns=[self.target_col]).columns
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        for ax, model, color in zip(
            axes, ["RandomForest", "LightGBM", "XGBoost"],
            ["skyblue", "lightgreen", "salmon"]):

            if model in self.feature_importances_:
                importances = self.feature_importances_[model]
                top_features = pd.Series(importances, index=feature_names).nlargest(10)
                ax.barh(top_features.index, top_features.values, color=color)
                ax.set_title(f"Top 10 Feature Importances - {model}")
                ax.invert_yaxis()

        plt.tight_layout()
        plt.show()
