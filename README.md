# 泰坦尼克号生存预测

用机器学习预测泰坦尼克号乘客是否幸存，对比了逻辑回归、决策树、随机森林和 XGBoost 四种模型。

## 数据集

- 来源：seaborn 内置泰坦尼克数据集
- 样本量：891 条，15 个字段
- 目标变量：`survived`（0 = 未幸存，1 = 幸存）

## 数据处理

- `age`：177 个缺失值，用中位数填充
- `embarked`：2 个缺失值，用众数填充
- `deck`：688 个缺失值，缺失率过高，直接丢弃
- `sex`：male → 0，female → 1

## 特征工程

在原始特征基础上构造了两个新特征：

- `family_size` = `sibsp` + `parch` + 1（家庭总人数）
- `is_alone`：家庭人数为 1 时为 1，否则为 0

最终使用的特征：`pclass`、`sex`、`age`、`sibsp`、`parch`、`fare`、`family_size`、`is_alone`、`embarked_Q`、`embarked_S`

## 模型与结果

单次划分（80/20）准确率：

| 模型 | 准确率 |
|---|---|
| LogisticRegression | 0.7989 |
| DecisionTree | 0.7989 |
| **RandomForest** | **0.8268** |
| XGBoost | 0.7989 |

5 折交叉验证：

| 模型 | 平均准确率 |
|---|---|
| LogisticRegression | 0.7935 (±0.0210) |
| DecisionTree | 0.7767 (±0.0349) |
| RandomForest | 0.8137 (±0.0314) |
| **XGBoost** | **0.8171 (±0.0337)** |

综合两项结果，**随机森林**在单次测试中表现最好（0.8268），XGBoost 在交叉验证中略高，两者差距在误差范围内。

## 特征重要性（随机森林）

| 特征 | 重要性 |
|---|---|
| sex | 0.269 |
| fare | 0.265 |
| age | 0.246 |
| pclass | 0.076 |
| family_size | 0.052 |
| 其他 | 0.092 |

**性别、票价、年龄**三个特征合计贡献约 78% 的预测能力。

## 文件说明

- `titanic_ml.ipynb`：完整代码，含数据处理、特征工程、模型训练与评估

## 环境

Python 3.6+，pandas、numpy、scikit-learn、xgboost、seaborn、matplotlib
