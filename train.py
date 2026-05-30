from sklearn.tree import DecisionTreeRegressor
from misc import load_data, split_data, evaluate

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = evaluate(y_test, preds)

    print("Decision Tree MSE:", mse)

if __name__ == "__main__":
    main()