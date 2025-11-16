
# machinelearning.py (corrigé)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.model_selection import train_test_split, learning_curve
from xgboost import XGBClassifier, plot_importance, plot_tree
from charger import charge
import streamlit as st
import joblib
import shap

# -----------------------------
# CACHING
# -----------------------------
@st.cache_data
def train_test():
    """
    Charge et prépare les données (balance, pdays transformés via charger.charge).
    Retour : x_train, x_test, y_train, y_test
    """
    x_train, x_test, y_train, y_test = charge.train_test(r"E:\fraude\donnees\bank.csv", "deposit")

    # Appliquer les mappings fournis dans charger
    for df in (x_train, x_test):
        if "balance" in df.columns:
            df["balance"] = df["balance"].map(charge.balance)
        if "pdays" in df.columns:
            df["pdays"] = df["pdays"].map(charge.pdays)

    return x_train, x_test, y_train, y_test


# Instanciation des données (appel unique au module)
x_train, x_test, y_train, y_test = train_test()


# -----------------------------
# EVALUATION
# -----------------------------
@st.cache_data
def feature():
    """Renvoie le dataframe de features (pour les selectbox du formulaire)."""
    return charge.feature(r"E:\fraude\donnees\bank.csv", "deposit")


def evaluation(model_base):
    """
    Entraîne le modèle `model_base` et affiche :
    - accuracy sur le train
    - classification report
    - matrice de confusion
    - learning curves (train / validation)
    - bouton importance des features (XGBoost)
    - bouton SHAP summary plot
    """
    # entraînement
    model = model_base.fit(x_train, y_train)
    st.text(f"Accuracy sur les données d'entraînement : {model.score(x_train, y_train):.4f}")

    # prédiction
    y_pred = model.predict(x_test)
    st.text(classification_report(y_test, y_pred))

    # matrice de confusion
    st.subheader("Matrice de confusion")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # learning curve
    st.subheader("Courbes d'apprentissage (F1)")
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        x_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="f1",
        n_jobs=-1,
        shuffle=True,
        random_state=42
    )

    # plot train
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(train_sizes, train_scores.mean(axis=1), marker='o', label="Train (F1)")
    ax1.fill_between(train_sizes,
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1),
                     alpha=0.15)
    ax1.set_xlabel("N exemples")
    ax1.set_ylabel("F1 score")
    ax1.legend()
    st.pyplot(fig1)
    plt.close(fig1)

    # plot validation
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(train_sizes, val_scores.mean(axis=1), marker='o', label="Validation (F1)")
    ax2.fill_between(train_sizes,
                     val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1),
                     alpha=0.15)
    ax2.set_xlabel("N exemples")
    ax2.set_ylabel("F1 score")
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

    # importance des colonnes (XGBoost)
    if st.button("Afficher importance des colonnes"):
        try:
            fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
            plot_importance(model, ax=ax_imp)
            st.pyplot(fig_imp)
            plt.close(fig_imp)
        except Exception as e:
            st.error(f"Impossible d'afficher l'importance : {e}")

    # SHAP summary plot
    if st.button("Afficher SHAP (summary)"):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_test)
            shap.summary_plot(shap_values, x_test, show=False)
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Erreur SHAP : {e}")


# -----------------------------
# PREDICTION (UI)
# -----------------------------
@st.cache_data
def _load_model(path=r"E:\fraude\xgboostx.joblib"):
    """Charge et met en cache le modèle pour éviter de le recharger à chaque prédiction."""
    return joblib.load(path)


def prediction(age, job, marital, education, default_val, balance_val,
               housing_val, loan_val, contact_val, day, month_val, duration,
               campaign, pdays_val, previous, poutcome_val):
    """
    Construis un vecteur en respectant l'ordre attendu et renvoie un message avec probas.
    ATTENTION : les valeurs passées à cette fonction doivent être déjà encodées
    (ou alors on encode ici via les fonctions ci-dessous).
    """
    model = _load_model()
    x = np.array([age, job, marital, education, default_val, balance_val,
                  housing_val, loan_val, contact_val, day, month_val, duration,
                  campaign, pdays_val, previous, poutcome_val]).reshape(1, -1)

    proba = model.predict_proba(x)[0]
    pred = int(model.predict(x)[0])

    if pred == 0:
        return f"Le client va faire son dépôt à temps (probabilité = {proba[0]:.4f})"
    else:
        return f"Le client va faire son dépôt en retard (probabilité = {proba[1]:.4f})"


# -----------------------------
# FONCTIONS D'ENCODAGE
# -----------------------------
# Mapping centralisé pour plus de robustesse
MAPPINGS = {
    "default": {"no": 0, "yes": 1},
    "loan": {"no": 0, "yes": 1},
    "housing": {"no": 0, "yes": 1},
    "marital": {"divorced": 0, "married": 1, "single": 2, "unknown": 2},
    "contact": {"cellular": 0, "telephone": 1, "unknown": 2},
    "education": {"primary": 0, "secondary": 1, "tertiary": 2, "unknown": 3},
    "poutcome": {"failure": 0, "other": 1, "success": 2, "unknown": 3},
    "job": {
        "admin.": 0, "blue-collar": 1, "entrepreneur": 2, "housemaid": 3,
        "management": 4, "retired": 5, "self-employed": 6, "services": 7,
        "student": 8, "technician": 9, "unemployed": 10, "unknown": 11
    },
    "month": {
        "apr": 0, "aug": 1, "dec": 2, "feb": 3, "jan": 4, "jul": 5,
        "jun": 6, "mar": 7, "may": 8, "nov": 9, "oct": 10, "sep": 11
    }
}


def _encode(mapping_name: str, value: str, default=0):
    """Encode value using MAPPINGS safely."""
    if value is None:
        return default
    return MAPPINGS.get(mapping_name, {}).get(str(value).strip(), default)


def default(val):
    return _encode("default", val, default=0)


def loan(val):
    return _encode("loan", val, default=0)


def housing(val):
    return _encode("housing", val, default=0)


def marital(val):
    return _encode("marital", val, default=2)


def contact(val):
    return _encode("contact", val, default=2)


def education(val):
    return _encode("education", val, default=3)


def poutcome(val):
    return _encode("poutcome", val, default=3)


def job(val):
    return _encode("job", val, default=11)


def month(val):
    return _encode("month", val, default=11)


def balance(val):
    """
    Wrapper sur charge.balance pour garder le même comportement qu'avant.
    """
    try:
        return charge.balance(val)
    except Exception:
        # si charge.balance échoue, essayer de forcer en int/float
        try:
            return float(val)
        except Exception:
            return 0.0


def pdays(val):
    """
    Wrapper sur charge.pdays.
    """
    try:
        return charge.pdays(val)
    except Exception:
        try:
            return int(val)
        except Exception:
            return -1
