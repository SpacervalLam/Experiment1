# 导入必要库
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
# 加载数据集
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# 数据预处理
def preprocess(df):
    # 填充缺失值
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

    # 创建新特征
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['IsAlone'] = (df['FamilySize'] == 0).astype(int)

    # 转换分类特征
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df = pd.get_dummies(df, columns=['Embarked'], prefix='Emb')

    return df


# 应用预处理
train_processed = preprocess(train_data)
test_processed = preprocess(test_data)

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'FamilySize', 'IsAlone', 'Emb_C', 'Emb_Q', 'Emb_S']

# 准备训练数据
X_train = train_processed[features]
y_train = train_data["Survived"]
X_test = test_processed[features]

# 构建随机森林模型
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    min_samples_split=6,
    random_state=42
)

# 训练与预测
model.fit(X_train, y_train)
predictions = model.predict(X_test)
predictions = predictions.astype(int)

# 生成提交文件
output = pd.DataFrame({
    'PassengerId': test_data.PassengerId,
    'Survived': predictions
})
output.to_csv('submission_demo2.csv', index=False)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices])
plt.xlim([-1, X_train.shape[1]])
plt.show()