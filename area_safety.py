import sys
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

#######################
# PREPROCESSING LOGIC #
#######################

def clean_column_names(df):
    df.columns = [c.strip() for c in df.columns]
    return df

def sanitize_filename(s):
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)

def drop_irrelevant_cols(df):
    drop_cols = ["Timestamp", "Email address"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

def convert_wine_shops(val):
    """Extract the number from wine shop entries."""
    try:
        if isinstance(val, str):
            if "more than" in val:
                return 6
            m = re.search(r"\d+", val)
            if m:
                return int(m.group())
        return 0
    except:
        return 0

def encode_yes_no(val):
    val = str(val).strip().lower()
    if val in ["yes", "yes, nearby", "yes, within the area"]:
        return 1
    elif val == "no":
        return 0
    else:
        return -1  # For "not sure" or unclear

def preprocess_area_safety(df):
    df = clean_column_names(df)
    df = drop_irrelevant_cols(df)

    # Wine shops as numeric
    if "Number of wine shops near to that area?" in df.columns:
        df["Number of wine shops near to that area?"] = df["Number of wine shops near to that area?"].apply(convert_wine_shops)

    # Yes/No/Not sure binary encoding
    yn_columns = [
        "Is this area known for criminal activity?",
        "Does the area have proper street lighting?",
        "Is this area is located to nearby police station?",
        "Is there a graveyard in or near the area?",
        "Do you consider this area to be safe?"
    ]
    for col in yn_columns:
        if col in df.columns:
            df[col] = df[col].apply(encode_yes_no)

    # Surveillance: one-hot
    if "What kind of surveillance is present in the area?" in df.columns:
        surveillance_dummies = df["What kind of surveillance is present in the area?"].str.get_dummies(sep=",")
        surveillance_dummies.columns = ["surveillance_" + c.strip().replace(" ", "_").lower() for c in surveillance_dummies.columns]
        df = pd.concat([df, surveillance_dummies], axis=1)
        df = df.drop(columns=["What kind of surveillance is present in the area?"])

    # Smuggling: multi-label one-hot
    if "Is the area known for any type of smuggling activity?" in df.columns:
        smuggle_types = ["drug smuggling", "alcohol smuggling", "human trafficking", "no known smuggling activity"]
        for t in smuggle_types:
            df["smuggle_" + t.replace(" ", "_")] = df["Is the area known for any type of smuggling activity?"].apply(
                lambda x: 1 if isinstance(x, str) and t in x.lower() else 0
            )
        df = df.drop(columns=["Is the area known for any type of smuggling activity?"])

    # For ALL remaining non-numeric columns except target, apply:
    # - If unique values ≤ 20, label encode
    # - Else, one-hot encode top 10 (group rest as "other")
    target_col = "Do you consider this area to be safe?"
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == "object":
            nunique = df[col].nunique()
            if nunique <= 20:
                df[col] = df[col].astype("category").cat.codes
            else:
                # One-hot encode top 10 values, group others as "other"
                top_vals = df[col].value_counts().nlargest(10).index
                df[col + "_grouped"] = df[col].apply(lambda x: x if x in top_vals else "other")
                dummies = pd.get_dummies(df[col + "_grouped"], prefix=sanitize_filename(col), drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col, col + "_grouped"])

    return df

##########################
# DESCRIPTIVE ANALYSIS   #
##########################

def descriptive_analysis(df):
    print("\n====== DESCRIPTIVE ANALYSIS ======\n")
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print(df.describe(include='all').T)
    print("\nValue counts for each column:\n")
    for col in df.columns:
        print(f"{col}:")
        print(df[col].value_counts())
        print()

##########################
# EDA WITH CHARTS        #
##########################

def eda_charts(df, output_dir="eda_charts"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nEDA charts will be saved in: {output_dir}/\n")

    # Numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        fname_hist = sanitize_filename(col) + "_hist.png"
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fname_hist}")
        plt.close()

        fname_box = sanitize_filename(col) + "_box.png"
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fname_box}")
        plt.close()

    # Correlation heatmap
    if len(num_cols) > 1:
        plt.figure(figsize=(10,8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png")
        plt.close()

    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        fname_bar = sanitize_filename(col) + "_bar.png"
        plt.figure()
        df[col].value_counts().head(10).plot(kind="bar")
        plt.title(f"Top Categories in {col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fname_bar}")
        plt.close()

    print("Charts saved. Review them for visual insights.\n")

##########################
# PREDICTIVE ANALYSIS    #
##########################

def predictive_analysis(df):
    print("\n====== PREDICTIVE ANALYSIS ======\n")
    target_col = "Do you consider this area to be safe?"
    if target_col not in df.columns:
        print("Target column not found. Skipping predictive analysis.")
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    print(f"Random Forest Classifier - Accuracy: {acc:.2f}\n")
    print(report)
    print("Confusion Matrix:\n", cm)

    # Feature importances
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    print("Top features influencing 'area safety' predictions:\n")
    print(feature_importances.sort_values(ascending=False).head(7))
    return feature_importances

##########################
# PRESCRIPTIVE ANALYSIS  #
##########################

def prescriptive_analysis(feature_importances):
    print("\n====== PRESCRIPTIVE ANALYSIS ======\n")
    high_impact = feature_importances.sort_values(ascending=False).head(3).index.tolist()
    msg = (
        f"Actionable Insights:\n"
        f"- The most influential features for predicting area safety are: {', '.join(high_impact)}.\n"
        f"- To improve safety, focus on improving these aspects in at-risk areas.\n"
        f"- For example, if 'Number of wine shops' or 'surveillance' features are high-impact, interventions here may yield better safety outcomes.\n"
    )
    print(msg)

##########################
# MAIN SCRIPT            #
##########################

def main():
    if len(sys.argv) != 2:
        print("Usage: python area_safety_analysis.py area_safety.csv")
        return
    fname = sys.argv[1]
    print(f"Loading data from {fname} ...\n")
    df = pd.read_csv(fname)

    # Preprocess
    df_processed = preprocess_area_safety(df)
    print("Preprocessing complete. Columns after preprocessing:\n")
    print(list(df_processed.columns))

    # Descriptive
    descriptive_analysis(df_processed)

    # EDA
    eda_charts(df_processed)

    # Predictive
    feature_importances = predictive_analysis(df_processed)

    # Prescriptive
    if feature_importances is not None:
        prescriptive_analysis(feature_importances)

if __name__ == "__main__":
    main()