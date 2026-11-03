import os
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


# ============================================================
# SETTINGS
# ============================================================

CATEGORICAL_COLUMNS = [
    "LT", "LI", "LM", "LR", "LL",
    "Left_Orientation",
    "Left_Hand_Position",
    "RT", "RI", "RM", "RR", "RL",
    "Right_Orientation",
    "Right_Hand_Position"
]

LABEL_COLUMN = "label"

MODEL_PARAMS = {
    "hidden_layer_sizes": (128, 64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": 32,
    "learning_rate_init": 0.001,
    "max_iter": 500,
    "random_state": 42,
    "early_stopping": False
}


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(path):
    path = path.strip().strip('"').strip("'")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"CSV file is empty:\n{path}"
        )

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"'label' column not found in:\n{path}"
        )

    # The one-hand alphabets C, I, L, O, U and V intentionally
    # contain missing values for the unused hand.
    #
    # Some categorical columns contain numeric values such as
    # 0/1/2, while orientation columns contain F/B/S.
    # Convert categorical columns to one consistent string type.
    #
    # Missing categorical values are represented as "Missing".
    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = (
                df[column]
                .where(df[column].notna(), "Missing")
                .astype(str)
            )

    return df


# ============================================================
# GET USERNAME FROM MERGED CSV FILENAME
# ============================================================

def get_username_from_csv(path):
    """
    Example:
        Sayandeep_merged_dataset.csv
        -> Sayandeep

        Abhirup_merged_dataset.csv
        -> Abhirup
    """
    filename = os.path.basename(path)
    stem = os.path.splitext(filename)[0]

    suffix = "_merged_dataset"

    if stem.lower().endswith(suffix):
        username = stem[:-len(suffix)]
    else:
        username = stem

    username = username.strip()

    if not username:
        username = "Unknown"

    # Remove characters that Windows does not allow in folder names.
    username = re.sub(r'[<>:"/\\|?*]', "_", username)

    return username


# ============================================================
# CREATE NEW RESULT FOLDER
# ============================================================

def create_result_folder(train_username, test_username):
    """
    First run:
        MLP_train(Sayandeep)_test(Abhirup)_result

    If that folder already exists, create:
        MLP_train(Sayandeep)_test(Abhirup)_result_1
        MLP_train(Sayandeep)_test(Abhirup)_result_2
        ...
    """

    base_name = (
        f"MLP_train({train_username})"
        f"_test({test_username})"
        f"_result"
    )

    # Keep results beside Merged_Dataset, not inside it.
    project_folder = os.path.dirname(
        os.path.dirname(os.path.abspath(
            # This value will be replaced by main using train path.
            os.path.join("Merged_Dataset", "dummy.csv")
        ))
    )

    # This helper only creates the name. The actual parent path is
    # passed from main below.
    return base_name


def make_unique_result_folder(parent_folder, base_name):
    """
    Create a fresh folder every time the program is run.

    First run uses the exact requested name.
    Later runs get _1, _2, _3, ... so existing results are never overwritten.
    """
    candidate = os.path.join(parent_folder, base_name)

    if not os.path.exists(candidate):
        os.makedirs(candidate)
        return candidate

    counter = 1

    while True:
        candidate = os.path.join(
            parent_folder,
            f"{base_name}_{counter}"
        )

        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate

        counter += 1


# ============================================================
# BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(df):
    feature_columns = [
        column
        for column in df.columns
        if column != LABEL_COLUMN
    ]

    categorical_columns = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in feature_columns
    ]

    numerical_columns = [
        column
        for column in feature_columns
        if column not in categorical_columns
    ]

    # Numerical landmark coordinates:
    # Missing values are intentional for the absent hand in
    # one-hand alphabets. Fill with median and standardize.
    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    # Categorical columns were converted to strings in load_dataset().
    # "Missing" is preserved as a real category.
    categorical_pipeline = Pipeline([
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "num",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_columns
        )
    ])

    return (
        preprocessor,
        numerical_columns,
        categorical_columns
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    output_path,
    train_username,
    test_username
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names))
    )

    plt.figure(figsize=(14, 12))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(
        f"ISL MLP - Confusion Matrix\n"
        f"Train: {train_username} | Test: {test_username}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



# ============================================================
# ADDITIONAL VISUALIZATIONS
# ============================================================

def save_per_class_accuracy(
    y_true,
    y_pred,
    class_names,
    output_path,
    train_username,
    test_username
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names))
    )

    class_totals = cm.sum(axis=1)
    class_correct = np.diag(cm)

    class_accuracy = np.divide(
        class_correct,
        class_totals,
        out=np.zeros_like(class_correct, dtype=float),
        where=class_totals != 0
    ) * 100

    accuracy_df = pd.DataFrame({
        "Alphabet": class_names,
        "Accuracy": class_accuracy
    })

    plt.figure(figsize=(16, 7))
    sns.barplot(
        data=accuracy_df,
        x="Alphabet",
        y="Accuracy"
    )

    plt.ylim(0, 100)
    plt.xlabel("ISL Alphabet")
    plt.ylabel("Accuracy (%)")
    plt.title(
        f"Per-Class Accuracy\n"
        f"Train: {train_username} | Test: {test_username}"
    )

    for index, value in enumerate(class_accuracy):
        plt.text(
            index,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def save_precision_recall_f1(
    y_true,
    y_pred,
    class_names,
    output_path,
    train_username,
    test_username
):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0
    )

    metrics_df = pd.DataFrame({
        "Alphabet": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    metrics_long = metrics_df.melt(
        id_vars="Alphabet",
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(18, 8))
    sns.barplot(
        data=metrics_long,
        x="Alphabet",
        y="Score",
        hue="Metric"
    )

    plt.ylim(0, 1)
    plt.xlabel("ISL Alphabet")
    plt.ylabel("Score")
    plt.title(
        f"Precision, Recall and F1-Score by Alphabet\n"
        f"Train: {train_username} | Test: {test_username}"
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def save_class_support(
    y_true,
    class_names,
    output_path,
    train_username,
    test_username
):
    counts = np.bincount(
        y_true,
        minlength=len(class_names)
    )

    support_df = pd.DataFrame({
        "Alphabet": class_names,
        "Test Samples": counts
    })

    plt.figure(figsize=(16, 7))
    sns.barplot(
        data=support_df,
        x="Alphabet",
        y="Test Samples"
    )

    plt.xlabel("ISL Alphabet")
    plt.ylabel("Number of Test Samples")
    plt.title(
        f"Test Sample Distribution by Alphabet\n"
        f"Train: {train_username} | Test: {test_username}"
    )

    for index, value in enumerate(counts):
        plt.text(
            index,
            value + max(1, max(counts) * 0.01),
            str(value),
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def save_training_loss_curve(
    model,
    output_path,
    train_username,
    test_username
):
    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "loss_curve_"):
        return

    if not classifier.loss_curve_:
        return

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x=np.arange(1, len(classifier.loss_curve_) + 1),
        y=classifier.loss_curve_,
        linewidth=2
    )

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(
        f"MLP Training Loss Curve\n"
        f"Train: {train_username} | Test: {test_username}"
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ISL MLP - TRAIN AND TEST ONLY")
    print("=" * 70)

    train_path = input(
        "\nEnter TRAINING CSV path: "
    ).strip().strip('"').strip("'")

    test_path = input(
        "Enter TESTING CSV path: "
    ).strip().strip('"').strip("'")

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    print(f"\nTraining rows: {len(train_df)}")
    print(f"Testing rows : {len(test_df)}")

    # --------------------------------------------------------
    # Verify feature columns
    # --------------------------------------------------------

    train_features = [
        column
        for column in train_df.columns
        if column != LABEL_COLUMN
    ]

    test_features = [
        column
        for column in test_df.columns
        if column != LABEL_COLUMN
    ]

    if train_features != test_features:

        missing_in_test = [
            column
            for column in train_features
            if column not in test_features
        ]

        extra_in_test = [
            column
            for column in test_features
            if column not in train_features
        ]

        raise ValueError(
            "\nTraining and testing feature columns do not match.\n"
            f"Missing in test: {missing_in_test}\n"
            f"Extra in test: {extra_in_test}"
        )

    # --------------------------------------------------------
    # Separate features and labels
    # --------------------------------------------------------

    X_train = train_df.drop(
        columns=[LABEL_COLUMN]
    )

    y_train_text = train_df[
        LABEL_COLUMN
    ].astype(str)

    X_test = test_df.drop(
        columns=[LABEL_COLUMN]
    )

    y_test_text = test_df[
        LABEL_COLUMN
    ].astype(str)

    # --------------------------------------------------------
    # Encode target labels
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    # Fit using training labels, which is the proper train/test
    # separation. Test labels are only checked against the
    # classes learned from training.
    y_train = label_encoder.fit_transform(
        y_train_text
    )

    unknown_test_labels = sorted(
        set(y_test_text)
        - set(label_encoder.classes_)
    )

    if unknown_test_labels:
        raise ValueError(
            "Testing data contains labels not present "
            f"in training data: {unknown_test_labels}"
        )

    y_test = label_encoder.transform(
        y_test_text
    )

    class_names = list(
        label_encoder.classes_
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    (
        preprocessor,
        numerical_columns,
        categorical_columns
    ) = build_preprocessor(train_df)

    print("\n" + "=" * 70)
    print("DATA PREPROCESSING")
    print("=" * 70)

    print(
        f"\nNumerical features  : "
        f"{len(numerical_columns)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_columns)}"
    )

    train_nan = int(
        X_train[numerical_columns].isna().sum().sum()
    )

    test_nan = int(
        X_test[numerical_columns].isna().sum().sum()
    )

    print(
        f"Missing numerical values in training data: "
        f"{train_nan}"
    )

    print(
        f"Missing numerical values in testing data : "
        f"{test_nan}"
    )

    print(
        "\nMissing values are handled inside the pipeline."
    )

    print(
        "The original CSV files are NOT modified."
    )

    # --------------------------------------------------------
    # Create model pipeline
    # --------------------------------------------------------

    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            MLPClassifier(
                **MODEL_PARAMS
            )
        )
    ])

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    print(
        f"\nMLP Architecture: "
        f"{MODEL_PARAMS['hidden_layer_sizes']}"
    )

    print(
        f"Activation       : "
        f"{MODEL_PARAMS['activation']}"
    )

    print(
        f"Solver           : "
        f"{MODEL_PARAMS['solver']}"
    )

    print(
        f"Alpha            : "
        f"{MODEL_PARAMS['alpha']}"
    )

    print(
        f"Batch size       : "
        f"{MODEL_PARAMS['batch_size']}"
    )

    print(
        f"Learning rate    : "
        f"{MODEL_PARAMS['learning_rate_init']}"
    )

    print(
        f"Maximum iterations: "
        f"{MODEL_PARAMS['max_iter']}"
    )

    model.fit(
        X_train,
        y_train
    )

    print("\nTraining completed.")

    # --------------------------------------------------------
    # Testing
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TESTING")
    print("=" * 70)

    y_pred = model.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        f"\nTest Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------------
    # Determine usernames
    # --------------------------------------------------------

    train_username = get_username_from_csv(
        train_path
    )

    test_username = get_username_from_csv(
        test_path
    )

    # Parent = same level as Merged_Dataset
    merged_dataset_folder = os.path.dirname(
        os.path.abspath(train_path)
    )

    project_folder = os.path.dirname(
        merged_dataset_folder
    )

    base_result_folder_name = (
        f"MLP_train({train_username})"
        f"_test({test_username})"
        f"_result"
    )

    output_folder = make_unique_result_folder(
        project_folder,
        base_result_folder_name
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    confusion_path = os.path.join(
        output_folder,
        "confusion_matrix.png"
    )

    save_confusion_matrix(
        y_test,
        y_pred,
        class_names,
        confusion_path,
        train_username,
        test_username
    )

    # --------------------------------------------------------
    # Save additional visualizations
    # --------------------------------------------------------

    per_class_accuracy_path = os.path.join(
        output_folder,
        "per_class_accuracy.png"
    )

    save_per_class_accuracy(
        y_test,
        y_pred,
        class_names,
        per_class_accuracy_path,
        train_username,
        test_username
    )

    precision_recall_f1_path = os.path.join(
        output_folder,
        "precision_recall_f1.png"
    )

    save_precision_recall_f1(
        y_test,
        y_pred,
        class_names,
        precision_recall_f1_path,
        train_username,
        test_username
    )

    class_support_path = os.path.join(
        output_folder,
        "test_sample_distribution.png"
    )

    save_class_support(
        y_test,
        class_names,
        class_support_path,
        train_username,
        test_username
    )

    training_loss_path = os.path.join(
        output_folder,
        "training_loss_curve.png"
    )

    save_training_loss_curve(
        model,
        training_loss_path,
        train_username,
        test_username
    )

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    report_path = os.path.join(
        output_folder,
        "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "ISL MLP - TRAIN AND TEST ONLY\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            f"Training Dataset: {train_path}\n"
        )

        file.write(
            f"Testing Dataset : {test_path}\n\n"
        )

        file.write(
            f"Training User: {train_username}\n"
        )

        file.write(
            f"Testing User : {test_username}\n\n"
        )

        file.write(
            f"Training rows: {len(train_df)}\n"
        )

        file.write(
            f"Testing rows : {len(test_df)}\n"
        )

        file.write(
            f"Test Accuracy: "
            f"{test_accuracy * 100:.4f}%\n\n"
        )

        file.write(
            "Classification Report\n"
        )

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            report
        )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    predictions_df = test_df.copy()

    predictions_df[
        "predicted_label"
    ] = label_encoder.inverse_transform(
        y_pred
    )

    predictions_path = os.path.join(
        output_folder,
        "test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        output_folder,
        "isl_mlp_model.joblib"
    )

    label_encoder_path = os.path.join(
        output_folder,
        "label_encoder.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        label_encoder,
        label_encoder_path
    )

    # --------------------------------------------------------
    # Save model configuration
    # --------------------------------------------------------

    config_path = os.path.join(
        output_folder,
        "model_configuration.txt"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "ISL MLP MODEL CONFIGURATION\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            f"Training User : {train_username}\n"
        )

        file.write(
            f"Testing User  : {test_username}\n\n"
        )

        for key, value in MODEL_PARAMS.items():

            file.write(
                f"{key}: {value}\n"
            )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        f"\nTrain User: {train_username}"
    )

    print(
        f"Test User : {test_username}"
    )

    print(
        f"\nResult folder:"
    )

    print(
        output_folder
    )

    print(
        f"\nModel              : "
        f"{model_path}"
    )

    print(
        f"Label encoder      : "
        f"{label_encoder_path}"
    )

    print(
        f"Confusion matrix   : "
        f"{confusion_path}"
    )

    print(
        f"Per-class accuracy : "
        f"{per_class_accuracy_path}"
    )

    print(
        f"Precision/Recall/F1: "
        f"{precision_recall_f1_path}"
    )

    print(
        f"Sample distribution: "
        f"{class_support_path}"
    )

    print(
        f"Training loss curve: "
        f"{training_loss_path}"
    )

    print(
        f"Classification     : "
        f"{report_path}"
    )

    print(
        f"Predictions        : "
        f"{predictions_path}"
    )

    print(
        f"Configuration      : "
        f"{config_path}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
