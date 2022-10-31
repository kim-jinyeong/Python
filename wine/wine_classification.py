import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class Wine:
    def solution(self):
        train = pd.read_csv('./data/train.csv').drop(['index'], axis=1)
        test = pd.read_csv('./data/test.csv').drop(['index'], axis=1)
        smpl_sub = pd.read_csv('./data/sample_submission.csv')

        # white 0, red 1
        train.loc[train['type'] == 'white', 'type'] = 0
        train.loc[train['type'] == 'red', 'type'] = 1

        test.loc[test['type'] == 'white', 'type'] = 0
        test.loc[test['type'] == 'red', 'type'] = 1

        x_data = train.drop(['quality'], axis=1)
        y_data = train['quality']
        x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_data, test_size=0.3, random_state=123, stratify=y_data)


        train_dataset = Pool(data=x_train,
                             label=y_train)

        eval_dataset = Pool(data=x_eval,
                            label=y_eval)

        model = CatBoostClassifier(task_type='GPU')

        model.fit(train_dataset, eval_set=eval_dataset, use_best_model=True, early_stopping_rounds=100, verbose=100)

        preds = model.predict(x_eval, prediction_type='Class')

        # 혼동행렬
        print(confusion_matrix(y_eval, preds))
        # 정확도
        print(accuracy_score(y_eval, preds))
        # 정밀도
        print(precision_score(y_eval, preds, average='macro'))
        # 재현율
        print(recall_score(y_eval, preds, average='macro'))
        # f1 score
        print(f1_score(y_eval, preds, average='macro'))

        # preds = model.predict(test, prediction_type='Class')
        #
        # smpl_sub['quality'] = preds
        # print(smpl_sub)
        # smpl_sub.to_csv('./data/submission.csv', index=False)




if __name__ == '__main__':
    Wine().solution()
