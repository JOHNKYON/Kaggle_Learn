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

    x_train = x_train.select_dtypes(exclude=['object'])
    x_test = x_test.select_dtypes(exclude=['object'])

    # Train-test split
    # pylint: disable=line-too-long
    # x_train, x_test, y_train, y_test = train_test_split(melb_numeric_predictors, melb_target,
    #                                                     train_size=0.7, test_size=0.3, random_state=0)

    # Impute data
    my_imputer = Imputer()

    col_with_missing = (col for col in x_train.columns
                        if x_train[col].isnull().any())

    for col in col_with_missing:
        x_train[col + '_was_missing'] = x_train[col].isnull()
        x_test[col + '_was_missing'] = x_test[col].isnull()

    x_train = my_imputer.fit_transform(x_train)
    x_test = my_imputer.fit_transform(x_test)

    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    print(preds)

    my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': preds})
    my_submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
