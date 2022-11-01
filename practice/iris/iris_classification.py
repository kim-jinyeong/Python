import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class iris():
    def test(self):
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        X = df.drop(['target'], axis=1)
        y = df.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        model = CatBoostClassifier(task_type='GPU')
        train_dataset = Pool(data=X_train,
                             label=y_train)

        eval_dataset = Pool(data=X_test,
                            label=y_test)

        model.fit(train_dataset, eval_set=eval_dataset, use_best_model=True, early_stopping_rounds=100, verbose=100)

        preds = model.predict(X_test, prediction_type='Class')

        # 혼동행렬
        print(confusion_matrix(y_test, preds))
        # 정확도
        print(accuracy_score(y_test, preds))
        # 정밀도
        print(precision_score(y_test, preds, average='macro'))
        # 재현율
        print(recall_score(y_test, preds, average='macro'))
        # f1 score
        print(f1_score(y_test, preds, average='macro'))
if __name__ == '__main__':
    iris().test()
