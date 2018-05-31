"""Python script for kaggle house price predict practice"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
import pandas as pd


def main():
    """Main script"""
    # Load data
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')

    y_train = train_data.SalePrice
    x_train = train_data.drop(['SalePrice'], axis=1)
    x_test = test_data

    # Encoding data
    x_train = pd.get_dummies(x_train)
    x_test = pd.get_dummies(x_test)
    x_train, x_test = x_train.align(x_test, join='left', axis=1)

    # Impute data
    my_imputer = Imputer()
    x_train = my_imputer.fit_transform(x_train)
    x_test = my_imputer.transform(x_test)
    print(x_train)

    # Get model
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    # Output
    submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': pred})
    submission.to_csv("hot_encoding/submission.csv", index=False)


if __name__ == '__main__':
    main()
