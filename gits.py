import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

print("读取数据...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"训练集大小: {train.shape}")
print(f"测试集大小: {test.shape}")

y = train['PitNextLap']
X = train.drop(columns=['PitNextLap', 'id'])
test_ids = test['id']
test_data = test.drop(columns=['id'])

print("开始特征工程...")

if 'TyreLife' in X.columns:
    X['TyreLife_square'] = X['TyreLife'] ** 2
    X['TyreLife_log'] = np.log1p(X['TyreLife'])
    X['TyreLife_sqrt'] = np.sqrt(X['TyreLife'])
    test_data['TyreLife_square'] = test_data['TyreLife'] ** 2
    test_data['TyreLife_log'] = np.log1p(test_data['TyreLife'])
    test_data['TyreLife_sqrt'] = np.sqrt(test_data['TyreLife'])

if 'LapNumber' in X.columns:
    X['Lap_ratio'] = X['LapNumber'] / (X['LapNumber'].max() + 1)
    X['Lap_square'] = X['LapNumber'] ** 2
    test_data['Lap_ratio'] = test_data['LapNumber'] / (test_data['LapNumber'].max() + 1)
    test_data['Lap_square'] = test_data['LapNumber'] ** 2

if 'TyreLife' in X.columns and 'LapNumber' in X.columns:
    X['Tyre_per_lap'] = X['TyreLife'] / (X['LapNumber'] + 1)
    test_data['Tyre_per_lap'] = test_data['TyreLife'] / (test_data['LapNumber'] + 1)

if 'Position' in X.columns:
    X['Is_leader'] = (X['Position'] == 1).astype(int)
    X['Is_top3'] = (X['Position'] <= 3).astype(int)
    test_data['Is_leader'] = (test_data['Position'] == 1).astype(int)
    test_data['Is_top3'] = (test_data['Position'] <= 3).astype(int)

print("数据预处理...")

numeric_cols = X.select_dtypes(include=[np.number]).columns
object_cols = X.select_dtypes(include=['object']).columns

print(f"数字列数量: {len(numeric_cols)}")
print(f"文字列数量: {len(object_cols)}")

X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
test_data[numeric_cols] = test_data[numeric_cols].fillna(test_data[numeric_cols].median())

for col in object_cols:
    if col in X.columns:
        mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown'
        X[col] = X[col].fillna(mode_val)
        if col in test_data.columns:
            test_data[col] = test_data[col].fillna(mode_val)

for col in object_cols:
    if col in X.columns and col in test_data.columns:
        all_values = pd.concat([X[col], test_data[col]], axis=0).unique()
        le = LabelEncoder()
        le.fit(all_values)
        X[col] = le.transform(X[col])
        test_data[col] = le.transform(test_data[col])

print(f"最终特征数量: {X.shape[1]}")

print("训练 LightGBM 模型...")

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)

model.fit(X, y)

print("预测中...")
pred = model.predict_proba(test_data)[:, 1]

sub = pd.DataFrame({'id': test_ids, 'PitNextLap': pred})
sub.to_csv('submission.csv', index=False)

print("完成！")
print("提交文件已保存: submission.csv")
print("Top 10 最重要的特征:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print(feature_importance.head(10))