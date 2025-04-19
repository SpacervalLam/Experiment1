import numpy as np
import pandas as pd


def process_df(df):
    """
    统一处理合并后的完整数据集（包含train和test）
    """
    df = df.copy()

    # ----------------- 特征工程步骤 -----------------
    # 设置索引
    df = df.set_index('PassengerId')

    # 提取Title
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # 标准化Title
    title_mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}
    df['Title'] = df['Title'].replace(title_mapping)

    # 处理罕见Title
    title_counts = df['Title'].value_counts()
    common_titles = title_counts[title_counts >= 10].index  # 出现次数>=10的视为常见Title
    title_mask = ~df['Title'].isin(common_titles)
    df.loc[title_mask, 'Title'] = df.loc[title_mask, 'Sex'].map({'male': 'Mr', 'female': 'Mrs'})

    # 填充Age（使用各Title的中位数）
    # 中位数是指将数据排序后处于正中间的那个数，能够减少异常值的影响，通常是一个更好的填补缺失值的选择。
    title_age_medians = df.groupby('Title')['Age'].median()
    print(title_age_medians)
    df['Age'] = df.groupby('Title')['Age'].transform(
        lambda x: x.fillna(title_age_medians[x.name])
    )

    # 填充Embarked和Fare
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)

    fare_median = df[df['Pclass'] == 3]['Fare'].median() # 3等舱的票价中位数
    df['Fare'] = df['Fare'].fillna(fare_median)

    # 交互特征
    df['Age*Class'] = df['Age'] * df['Pclass']
    df['Age*Fare'] = df['Age'] * df['Fare']

    # 独热编码（确保列一致）
    categorical_cols = ['Sex', 'Pclass', 'Embarked', 'Title']
    df_encoded = pd.get_dummies(df[categorical_cols],
                                columns=categorical_cols,
                                prefix_sep='_',
                                drop_first=False,  # 保留所有类别
                                dummy_na=False)  # 忽略缺失值

    # 家庭特征
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 分箱处理
    age_bins = [0, 12, 20, 40, 60, np.inf]
    df['AgeBand'] = pd.cut(df['Age'], bins=age_bins, labels=range(5)).astype(int)

    fare_q = df['Fare'].quantile([0.25, 0.5, 0.75]).tolist()
    fare_bins = [-np.inf] + fare_q + [np.inf]
    df['FareBand'] = pd.cut(df['Fare'], bins=fare_bins, labels=range(4)).astype(int)

    # 对数变换
    df['Fare_log'] = np.log1p(df['Fare'])

    # HasCabin
    df['HasCabin'] = df['Cabin'].notnull().astype(int)

    # 合并所有特征
    df = pd.concat([df, df_encoded], axis=1)

    # 删除冗余列
    cols_to_drop = ['Name', 'Ticket', 'Cabin', 'Sex', 'Pclass', 'Embarked', 'Title', 'Fare', 'SibSp', 'Parch']
    df = df.drop(cols_to_drop, axis=1, errors='ignore')

    # 标准化（使用全部数据集均值和标准差）
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if 'Survived' in numeric_cols:
        numeric_cols.remove('Survived')  # 排除标签列

    # 计算全部数据集统计量
    mew = df[numeric_cols].mean()
    std = df[numeric_cols].std()

    # 应用标准化
    df[numeric_cols] = (df[numeric_cols] - mew) / std


    return df


# ----------------- 主流程 -----------------
# 读入数据集
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 合并数据集
full_data = pd.concat([train, test], sort=False)

# 统一处理数据
full_data = process_df(full_data)

# 拆分回训练集和测试集
X_train = full_data.loc[train['PassengerId']].drop('Survived', axis=1)
y_train = train['Survived']
X_test = full_data.loc[test['PassengerId']].drop('Survived', axis=1)

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
    score=cross_val_score(model, X_train, train['Survived'], cv=5)
    scores.append(score.mean())
    print("{}:{},{}".format(name,score.mean(),score))


# 训练模型
from sklearn.ensemble import RandomForestClassifier

# Load model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    min_samples_split=6,
    random_state=42
)

model.fit(X_train, y_train)

# 预测结果
y_pred = model.predict(X_test)

# 保存结果
submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred})
submission.to_csv('submission_demo3.csv', index=False)
print("Saved submission to submission_demo3.csv")

import matplotlib.pyplot as plt

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices])
plt.xlim([-1, X_train.shape[1]])
plt.xticks(rotation=45, ha='right')
plt.show()