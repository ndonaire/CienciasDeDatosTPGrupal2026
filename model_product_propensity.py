import json
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
TARGET = "target_product_acquired"
ID_COL = "customer_id"
TRAIN_FILE = Path("bank_product_propensity_dataset.csv")
PREDICT_FILE = Path("bank_product_propensity_predict.csv")
PREDICTIONS_FILE = Path("bank_product_propensity_predictions.csv")
METRICS_FILE = Path("model_validation_metrics.json")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    snapshot_date = pd.to_datetime(df["snapshot_date"])
    df["snapshot_month"] = snapshot_date.dt.month
    df["snapshot_dayofyear"] = snapshot_date.dt.dayofyear

    # Variables de balance y flujos monetarios
    df["total_liquid_balance"] = df[["savings_balance", "checking_balance"]].sum(
        axis=1, min_count=1
    )
    df["total_financial_balance"] = df[
        ["savings_balance", "checking_balance", "investment_balance"]
    ].sum(axis=1, min_count=1)
    df["deposit_withdrawal_net"] = (
        df["total_deposit_amount_last_month"]
        - df["total_withdrawal_amount_last_month"]
    )
    
    # AJUSTE: Nombres reales de logins según el dataset mapeado
    df["digital_logins_total"] = (
        df["num_mobile_logins_last_month"] + df["num_web_logins_last_month"]
    )

    # Flags e indicadores lógicos de negocio adaptados a las columnas reales
    df["has_any_investment_signal"] = (
        (df["investment_balance"] > 0)
        | (df["has_investment_account"] == 1)
    ).astype(int)

    # AJUSTE: Nombre real de utilización de tarjeta de crédito
    df["has_active_credit_products"] = (
        (df["credit_utilization"] > 0)
        | (df["has_personal_loan"] == 1)
    ).astype(int)

    # Interacciones NO lineales para mejorar el rendimiento de la Regresión Logística
    df["debt_by_risk_interaction"] = df["debt_to_income_ratio"] * df["risk_score"]
    
    # Indicador booleano (binning) de perfil financiero premium
    df["premium_financial_profile"] = (
        (df["credit_score"] > 700) & (df["debt_to_income_ratio"] < 0.3)
    ).astype(int)

    # Estabilidad financiera recalculada con combinaciones cruzadas
    df["financial_stability"] = (
        df["credit_score"] 
        - (150 * df["debt_to_income_ratio"]) 
        + (50 * df["premium_financial_profile"])
    )

    return df


def group_split(
    df: pd.DataFrame, train_ratio: float = 0.8, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_customers = df[ID_COL].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_customers)

    train_size = int(len(unique_customers) * train_ratio)
    train_customers = set(unique_customers[:train_size])

    train_mask = df[ID_COL].isin(train_customers)
    return df[train_mask].reset_index(drop=True), df[~train_mask].reset_index(
        drop=True
    )


def make_design_matrices(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None = None,
    predict_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    
    drop_cols = [ID_COL, "snapshot_date", TARGET]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    # Identificar variables numéricas
    num_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Estadísticas para imputación y escalado calculadas únicamente con datos de entrenamiento
    medians = train_df[num_cols].median()
    means = train_df[num_cols].mean()
    stds = train_df[num_cols].std().replace(0, 1.0)

    def process_matrix(df: pd.DataFrame, is_train: bool = False) -> np.ndarray:
        X_processed = []
        for col in num_cols:
            s = df[col].copy()
            # Guardamos el flag indicador de faltante
            missing_col = s.isna().astype(float)
            
            # Imputación con mediana de train
            s = s.fillna(medians[col])
            
            # Escalado estándar (Z-score) basado en train
            s_scaled = (s - means[col]) / stds[col]

            X_processed.append(s_scaled.values)
            X_processed.append(missing_col.values)

        X_mat = np.column_stack(X_processed)
        # Término de sesgo (intercepto)
        bias = np.ones((X_mat.shape[0], 1))
        return np.hstack([bias, X_mat])

    X_train = process_matrix(train_df, is_train=True)
    y_train = train_df[TARGET].values if TARGET in train_df else np.array([])

    X_valid = process_matrix(valid_df) if valid_df is not None else None
    y_valid = valid_df[TARGET].values if valid_df is not None and TARGET in valid_df else None

    X_predict = process_matrix(predict_df) if predict_df is not None else None

    return X_train, y_train, X_valid, y_valid, X_predict


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    lr: float = 0.01,
    epochs: int = 3500,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    np.random.seed(seed)
    n_samples, n_features = X.shape
    w = np.random.normal(0, 0.01, size=n_features)

    # Inicialización del optimizador Adam manual
    m = np.zeros(n_features)
    v = np.zeros(n_features)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    for epoch in range(1, epochs + 1):
        # Descenso sutil del learning rate para asegurar convergencia perfecta en el óptimo
        current_lr = lr if epoch < (epochs * 0.8) else lr * 0.1
        
        p = sigmoid(X @ w)
        # Regularización L2 (se omite penalizar la columna 0 del intercepto)
        reg_term = alpha * w
        reg_term[0] = 0.0
        grad = (X.T @ (p - y)) / n_samples + reg_term

        # Momentos de Adam
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad**2)

        m_hat = m / (1.0 - beta1**epoch)
        v_hat = v / (1.0 - beta2**epoch)

        w -= current_lr * m_hat / (np.sqrt(v_hat) + eps)

    return w


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    desc_score_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[desc_score_indices]

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return {"auc_roc": 0.0, "f1_score_at_0.5": 0.0}

    # CORRECCIÓN: Cálculo manual del AUC-ROC moderno (sin usar np.trapz)
    # Calculamos el área bajo la curva sumando los rectángulos de los Verdaderos Positivos
    tp_cum = np.cumsum(y_true_sorted)
    auc = float(np.sum(tp_cum[y_true_sorted == 0]) / (n_pos * n_neg))

    # Métricas estándar en umbral de decisión 0.50
    y_pred = (y_prob >= 0.50).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "auc_roc": auc,
        "precision_at_0.5": precision,
        "recall_at_0.5": recall,
        "f1_score_at_0.5": f1,
    }


def extract_feature_importance(w: np.ndarray, train_df: pd.DataFrame) -> list[dict]:
    drop_cols = [ID_COL, "snapshot_date", TARGET]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    expanded_names = ["intercept"]
    num_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    for col in num_cols:
        expanded_names.append(col)
        expanded_names.append(f"{col}_missing")

    importance = []
    for name, weight in zip(expanded_names, w):
        importance.append(
            {
                "feature": name,
                "coefficient": float(weight),
                "absolute_coefficient": float(np.abs(weight)),
            }
        )

    return sorted(importance, key=lambda x: x["absolute_coefficient"], reverse=True)


def train_final_and_predict(
    train_df: pd.DataFrame, predict_df: pd.DataFrame, alpha: float, lr: float, epochs: int
) -> tuple[pd.DataFrame, list[dict]]:
    
    train_df_proc = add_features(train_df)
    predict_df_proc = add_features(predict_df)

    X_train, y_train, _, _, X_predict = make_design_matrices(
        train_df_proc, predict_df=predict_df_proc
    )

    # Ajuste final con el set completo pre-producción
    w_final = fit_logistic_regression(X_train, y_train, alpha=alpha, lr=lr, epochs=epochs)
    probabilities = sigmoid(X_predict @ w_final)

    output = pd.DataFrame({
        ID_COL: predict_df[ID_COL],
        "snapshot_date": predict_df["snapshot_date"],
        "probability_product_acquired": probabilities
    })

    # AJUSTE: Asignación limpia de deciles del 1 (mayor propensión) al 10 (menor propensión)
    output["propensity_decile"] = pd.qcut(
        probabilities,
        q=10,
        labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        duplicates="drop"
    ).astype(int)

    final_importance = extract_feature_importance(w_final, train_df_proc)
    return output, final_importance


def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    predict_df = pd.read_csv(PREDICT_FILE)

    #print("Columnas reales en el dataset:", train_df.columns.tolist())

    missing_train = train_df.isna().mean().sort_values(ascending=False)

    print("Running validation split...")
    train_split, valid_split = group_split(train_df, train_ratio=0.8)

    train_split_feat = add_features(train_split)
    valid_split_feat = add_features(valid_split)

    X_train, y_train, X_valid, y_valid, _ = make_design_matrices(
        train_split_feat, valid_df=valid_split_feat
    )

    # Hiperparámetros lógicos seleccionados
    alpha_opt = 0.05
    lr_opt = 0.02
    epochs_opt = 3500

    print("Training baseline model (Validation set)...")
    w_baseline = fit_logistic_regression(
        X_train, y_train, alpha=alpha_opt, lr=lr_opt, epochs=epochs_opt
    )
    val_prob_baseline = sigmoid(X_valid @ w_baseline)
    baseline_metrics = evaluate_predictions(y_valid, val_prob_baseline)
    baseline_importance = extract_feature_importance(w_baseline, train_split_feat)

    print("Evaluating validation metrics on final features...")
    w_valid = fit_logistic_regression(
        X_train, y_train, alpha=alpha_opt, lr=lr_opt, epochs=epochs_opt
    )
    val_prob_final = sigmoid(X_valid @ w_valid)
    final_metrics = evaluate_predictions(y_valid, val_prob_final)
    validation_importance = extract_feature_importance(w_valid, train_split_feat)

    print("Generating final predictions on target clients...")
    predictions, final_importance = train_final_and_predict(
        train_df, predict_df, alpha=alpha_opt, lr=lr_opt, epochs=epochs_opt
    )

    predictions.to_csv(PREDICTIONS_FILE, index=False)

    metrics_payload = {
        "dataset_shapes": {
            "train_rows": int(len(train_df)),
            "predict_rows": int(len(predict_df)),
        },
        "target_distribution": {
            str(k): int(v) for k, v in train_df[TARGET].value_counts().sort_index().items()
        },
        "target_positive_rate": float(train_df[TARGET].mean()),
        "missing_train_top_percent": {
            k: float(v) for k, v in missing_train.to_dict().items()
        },
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "baseline_importance": baseline_importance,
        "validation_importance": validation_importance,
        "final_importance_full_training": final_importance,
        "prediction_summary": {
            "rows": int(len(predictions)),
            "mean_probability": float(predictions["probability_product_acquired"].mean()),
            "min_probability": float(predictions["probability_product_acquired"].min()),
            "max_probability": float(predictions["probability_product_acquired"].max()),
            "top_decile_mean_probability": float(
                predictions.loc[
                    predictions["propensity_decile"] == 1,
                    "probability_product_acquired",
                ].mean()
            ),
        },
    }
    # AJUSTE: Sintaxis limpia para codificación en Windows sin barras extrañas
    METRICS_FILE.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(json.dumps(metrics_payload["final_metrics"], indent=2))
    print(f"Saved predictions to {PREDICTIONS_FILE}")
    print("Execution completed successfully.")


if __name__ == "__main__":
    main()