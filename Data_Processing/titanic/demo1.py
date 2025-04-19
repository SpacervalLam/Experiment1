import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"NumPy {np.__version__} imported" )
print(f"Pandas {pd.__version__} imported" )

# 读入数据集
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
# 复制数据避免修改原始数据
train_data = train.copy()
test_data = test.copy()
# 合并测试集和训练集以便于从全部数据总结特征
full_data = pd.concat([train_data,test_data],ignore_index=True)

# 删除不需要的列
full_data = full_data.drop(['Name', 'Ticket'], axis=1)

# 用众数填充缺失的Embarked
full_data.fillna({'Embarked': full_data['Embarked'].mode()[0]}, inplace=True)
full_data = pd.get_dummies(full_data, columns=['Embarked'], prefix='Embarked')

# 用平均值填充缺失的Fare
full_data.fillna({'Fare': full_data[full_data['Pclass'] == 3]['Fare'].mean()}, inplace=True)

from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0), # 回归模型
    max_iter=10, # 最大迭代次数
    random_state=42, # 随机数种子
    verbose=2 # 显示日志
)

# 填充Age列中的缺失值
imputer.fit(full_data[['Age']])
full_data['Age'] = imputer.transform(full_data[['Age']])

# 将Cabin转化为分类变量
full_data.loc[full_data['Cabin'].notnull(), 'Cabin'] = 1
full_data.loc[full_data['Cabin'].isnull(), 'Cabin'] = 0

# 处理Sex列
full_data['Sex'] = full_data['Sex'].map({'male': 0, 'female': 1})

# 创建新特征
full_data['FamilySize'] = full_data['SibSp'] + full_data['Parch'] + 1
full_data['IsAlone'] = 0
full_data.loc[full_data['FamilySize'] == 1, 'IsAlone'] = 1

# 根据Survived列将数据重新分为训练集和测试集
train_data = full_data[full_data['Survived'].notnull()]
test_data = full_data[full_data['Survived'].isnull()]

# 预测目标变量
from sklearn.preprocessing import StandardScaler # 标准化
scaler = StandardScaler()

train_data_X_scaled = scaler.fit(train_data.drop('Survived', axis=1)).transform(train_data.drop('Survived', axis=1))
train_data_X_scaled = pd.DataFrame(train_data_X_scaled, columns=train_data.drop('Survived', axis=1).columns)

test_data_X_scaled = scaler.fit(train_data.drop('Survived', axis=1)).transform(test_data.drop('Survived', axis=1))
test_data_X_scaled = pd.DataFrame(test_data_X_scaled, columns=test_data.drop('Survived', axis=1).columns)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score


models = [
    KNeighborsClassifier(3),
    LogisticRegression(random_state=0),
    GaussianNB(),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(random_state=0),
    GradientBoostingClassifier(random_state=0),
    SVC(kernel="linear", C=0.025)
]

names=['KNN','LR','NB','Tree','RF','GDBT','SVM']
scores=[]
for name, model in zip(names, models):
    score=cross_val_score(model, train_data_X_scaled, train_data['Survived'], cv=5)
    scores.append(score.mean())
    print("{}:{},{}".format(name,score.mean(),score))

# 选择模型
choice = str(input("请选择模型："))
if choice == 'RF':
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        min_samples_split=6,
        random_state=42
    )
elif choice == 'LR':
    model = LogisticRegression(random_state=0)
elif choice == 'KNN':
    model = KNeighborsClassifier(3)
elif choice == 'Tree':
    model = DecisionTreeClassifier(random_state=0)
elif choice == 'NB':
    model = GaussianNB()
elif choice == 'SVM':
    model = SVC(kernel="linear", C=0.025)
elif choice == 'GDBT':
    model = GradientBoostingClassifier(random_state=0)
else:
    print("输入错误，请重新输入")
    exit()
# 训练模型
model.fit(train_data_X_scaled, train_data['Survived'])

# 预测测试集
predictions = model.predict(test_data_X_scaled)
# float64转int
predictions = predictions.astype(int)

# 保存结果
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('submission.csv', index=False)
print("保存结果：submission_demo1.csv")
# 可视化特征重要性
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))
plt.title("Feature Importances")
plt.bar(range(train_data_X_scaled.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(train_data_X_scaled.shape[1]), train_data_X_scaled.columns[indices])
plt.xlim([-1, train_data_X_scaled.shape[1]])
plt.show()
