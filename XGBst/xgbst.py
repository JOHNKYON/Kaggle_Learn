"""Python script for kaggle house price predict practice"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
import pandas as pd


def main():
    """Main script"""
    # Load data
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')

    # Preprocessing data
    train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
    origin_train_y = train_data.SalePrice
    origin_train_x = train_data.drop(['SalePrice'], axis=1).select_dtypes(exclude='object')
    test_x = test_data.select_dtypes(exclude='object')

    # Encoding
    origin_train_x = pd.get_dummies(origin_train_x)
    test_x = pd.get_dummies(test_x)
    origin_train_x, x_test = origin_train_x.align(test_x, join='left', axis=1)

    # Impute data
    my_imputer = Imputer()
    origin_train_x = my_imputer.fit_transform(origin_train_x)
    test_x = my_imputer.transform(test_x)

    # pylint: disable=line-too-long
    train_x, cv_x, train_y, cv_y = train_test_split(origin_train_x, origin_train_y, test_size=0.25)

    my_model = XGBRegressor(n_estimators=100000, learning_rate=0.001)
    # Add verbose=False to avoid printing out updates with each cycle
    my_model.fit(train_x, train_y, early_stopping_rounds=10, eval_set=[(cv_x, cv_y)], verbose=True)
    best_iteration = my_model.best_iteration

    # Rout 1: Do not retrain the model

    # Make predictions
    # predictions = my_model.predict(test_x, ntree_limit=best_iteration)
    # print(predictions)

    # Rout 2: Re-train the model
    my_model = XGBRegressor(n_estimators=best_iteration, learning_rate=0.001)
    my_model.fit(origin_train_x, origin_train_y)

    predictions = my_model.predict(test_x)

    my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})
    my_submission.to_csv('XGBst/submission.csv', index=False)


if __name__ == "__main__":
    main()
