"""Train, compare and save Ra prediction models."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from common import CAT_COLS, NUM_COLS, RAW_COLS, TARGET, add_features


def make_pre():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])


def wrap(reg):
    """Preprocessing + regressor, trained on log(Ra) (Ra is positive & skewed)."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("pre", make_pre()), ("reg", reg)]),
        func=np.log, inverse_func=np.exp)


def get_models():
    models = {
        "Linear Regression": wrap(LinearRegression()),
        "RSM (Quadratic)": wrap(Pipeline([("poly", PolynomialFeatures(2)), ("r", Ridge(1.0))])),
        "SVR": wrap(SVR(C=10, epsilon=0.01, gamma="scale")),
        "Random Forest": wrap(RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        "Gradient Boosting": wrap(GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                                            max_depth=3, random_state=42)),
        "ANN (MLP)": wrap(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000, early_stopping=True,
                                       random_state=42)),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = wrap(XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                                              random_state=42))
    except Exception:
        pass
    return models


def metrics(y, p):
    return {"R2": r2_score(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": mean_absolute_error(y, p),
            "MAPE_%": float(np.mean(np.abs((y - p) / y)) * 100)}


def run(csv_path="data/milling_data.csv", out_dir="artifacts"):
    out = Path(out_dir); out.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(csv_path).dropna(subset=RAW_COLS + [TARGET])
    df = df[df[TARGET] > 0]
    X, y = add_features(df[RAW_COLS]), df[TARGET].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    rows, fitted = [], {}
    for name, mdl in get_models().items():
        cv = cross_validate(mdl, X_tr, y_tr, cv=5,
                            scoring=("r2", "neg_root_mean_squared_error"))
        mdl.fit(X_tr, y_tr)
        m = metrics(y_te, mdl.predict(X_te))
        m.update({"Model": name,
                  "CV_R2": cv["test_r2"].mean(),
                  "CV_RMSE": -cv["test_neg_root_mean_squared_error"].mean()})
        rows.append(m); fitted[name] = mdl
        print(f"{name:20s} R2={m['R2']:.4f} RMSE={m['RMSE']:.4f} MAPE={m['MAPE_%']:.2f}%")

    res = (pd.DataFrame(rows)[["Model", "R2", "RMSE", "MAE", "MAPE_%", "CV_R2", "CV_RMSE"]]
           .sort_values("CV_RMSE").reset_index(drop=True))
    best_name = res.loc[0, "Model"]
    best = fitted[best_name]

    # --- plots / importance (on held-out data, using the model fitted on train)
    pred = best.predict(X_te)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_te, pred, s=14, alpha=.6)
    lim = [0, max(y_te.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, "r--"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Measured Ra (um)"); ax.set_ylabel("Predicted Ra (um)")
    ax.set_title(f"{best_name}: predicted vs measured"); fig.tight_layout()
    fig.savefig(out / "pred_vs_actual.png", dpi=130); plt.close(fig)

    imp = permutation_importance(best, X_te, y_te, n_repeats=10, random_state=42,
                                 scoring="neg_root_mean_squared_error")
    imp_s = pd.Series(imp.importances_mean, index=X_te.columns).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    imp_s.plot.barh(ax=ax); ax.set_xlabel("Increase in RMSE when shuffled")
    ax.set_title("Permutation feature importance"); fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=130); plt.close(fig)

    # --- deploy: refit best model on ALL data
    final = get_models()[best_name].fit(X, y)
    joblib.dump(final, out / "model.joblib")
    res.to_csv(out / "model_comparison.csv", index=False)
    meta = {"best_model": best_name, "n_samples": int(len(df)),
            "ra_min": float(y.min()), "ra_max": float(y.max()),
            "importance": imp_s.sort_values(ascending=False).to_dict()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\nBest model:", best_name)
    return res


if __name__ == "__main__":
    run()
