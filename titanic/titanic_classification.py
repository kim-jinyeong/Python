import numpy as np
import pandas as pd
import hyperopt
from catboost import Pool, CatBoostClassifier, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Titanic:
    def __init__(self):
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()

    def rd_file(self):
        self.train = pd.read_csv('./data/train.csv')
        self.test = pd.read_csv('./data/test.csv')

    def solution(self):
        train = self.train

        # 데이터 확인
        print(train.info())

        '''
         0   PassengerId  탑승객의 고유 아이디
         1   Survived     생존여부(0: 사망, 1: 생존)
         2   Pclass       등실의 등급(1: 1등급, 2: 2등급, 3: 3등급)
         3   Name         이름
         4   Sex          성별
         5   Age          나이
         6   SibSp        함께 탑승한 형제자매, 아내 남편의 수
         7   Parch        함께 탑승한 부모, 자식의 수
         8   Ticket       티켓번호
         9   Fare         티켓의 요금
         10  Cabin        객실번호
         11  Embarked     배에 탑승한 위치 (C = Cherbourg, Q = Queenstown, S = Southampton)
        '''

        # 데이터 null값 확인
        print(train.isnull().sum())

        # 결측치 제거
        train.fillna(-1, inplace=True)


        X = train.drop(['Survived'], axis=1)
        y = train['Survived']

        cate_features_index = np.where(X.dtypes != float)[0]

        print(cate_features_index)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=123)

        model = CatBoostClassifier(task_type='GPU', verbose=100)

        model.fit(X_train, y_train, cat_features=cate_features_index, eval_set=(X_test, y_test))

        print('the test accuracy is :{:.6f}'.format(accuracy_score(y_test, model.predict(X_test))))

        return model

    def save(self, model):
        test = self.test

        test.fillna(-1, inplace=True)

        pred = model.predict(test)
        pred = pred.astype(np.int32)
        submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred})

        submission.to_csv('./data/catboost.csv', index=False)
if __name__ == '__main__':
    Titanic = Titanic()
    Titanic.rd_file()
    model = Titanic.solution()
    Titanic.save(model)