# F1 Pit Stop Prediction

基于 LightGBM 的 F1 进站时机预测模型 - Kaggle 游乐场系列赛 S6E5

## 项目简介

使用 F1 赛事数据预测赛车是否在下一圈进站（PitNextLap），评估指标为 ROC-AUC。

## 文件结构

## 环境要求

- Python 3.8+
- pandas
- numpy
- lightgbm
- scikit-learn

## 安装依赖

```bash
pip install pandas numpy lightgbm scikit-learn
python submit.py

特征列表
特征名称	说明
TyreLife_square	轮胎寿命平方
TyreLife_log	轮胎寿命对数
TyreLife_sqrt	轮胎寿命开方
Lap_ratio	圈数占比
Lap_square	圈数平方
Tyre_per_lap	每圈轮胎磨损
Is_leader	是否领先
Is_top3	是否前三
模型参数
算法：LightGBM

n_estimators：500

learning_rate：0.05

输出格式
id	PitNextLap
439140	0.7234
439141	0.2156
