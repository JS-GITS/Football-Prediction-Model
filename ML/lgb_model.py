import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss
from data_loader import data_loader

def lgb_model():
    train_df, valid_df, encoders = data_loader()

    # Define the parameters the model is going to be trained on
    feature_cols = [
        "Division_enc", "HomeTeam_enc", "AwayTeam_enc",
        "HomeElo", "AwayElo",
        "Form3Home", "Form5Home", "Form3Away", "Form5Away",
        "OddHome", "OddDraw", "OddAway",
        "HandiSize", "HandiHome", "HandiAway",
        "EloDiff", "EloSum", "EloFormRatio",
        "Form3Diff", "Form5Diff",
        "ProbHome", "ProbDraw", "ProbAway", "ProbDiff", "BookerMargin",
        "OddHome_missing", "OddDraw_missing", "OddAway_missing",
        "HandiSize_missing", "HandiHome_missing", "HandiAway_missing",
    ]

    categorical_feats = ["Division_enc", "HomeTeam_enc", "AwayTeam_enc"]

    x_train = train_df[feature_cols]
    y_train = train_df["Target"]

    x_test = valid_df[feature_cols]
    y_test = valid_df["Target"]

    # Define the LightGBM (Light Gradient Boosting Machine)
    model = lgb.LGBMClassifier(
        num_leaves=64,
        max_depth=-1,
        learning_rate=0.05,
        n_estimators=5000,
        subsample=0.9,
        colsample_bytree=0.9,
        min_data_in_leaf=30,
        reg_alpha=0.3,
        reg_lambda=1.5,
        objective="multiclass",
        num_class=3,
        random_state=42,
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)],
        categorical_feature=categorical_feats,
    )

    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Log loss:", log_loss(y_test, y_pred_proba))
    return model