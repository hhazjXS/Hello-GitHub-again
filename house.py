# 忽略警告
import warnings
warnings.filterwarnings('ignore')

# 引入基础包
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# 设置属性防止画图乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 绘图函数
def figure(title, *datalist):
    plt.figure(facecolor='white', figsize=[16, 8])
    for v in datalist:
        plt.plot(v[0], '-', label=v[1], linewidth=2)
        plt.plot(v[0], 'o')
    plt.grid()
    plt.title(title, fontsize=20)
    plt.legend(fontsize=16)
    plt.show()

# 加载房价的数据集
houses = pd.read_csv('housing.csv')
#print(houses)
houses_df = pd.DataFrame(houses._data, columns=['RM', 'LSTAT', 'PTRATIO', 'MEDV'])

#三个特征值和目标值之间的关系
# RM 和房价的散点图
plt.figure(facecolor='white')
plt.scatter(houses_df['RM'], houses_df['MEDV'], s=30, edgecolor='white')
plt.title('RM与MEDV的关系图')
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.show()

# LSTAT 和房价的散点图
plt.figure(facecolor='white')
plt.scatter(houses_df['LSTAT'], houses_df['MEDV'], s=30, edgecolor='white')
plt.title('LSTAT与MEDV的关系图')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')

# PTRATIO 和房价的散点图
plt.figure(facecolor='white')
plt.scatter(houses_df['PTRATIO'], houses_df['MEDV'], s=30, edgecolor='white')
plt.title('PTRATIO与MEDV的关系图')
plt.xlabel('PTRATIO')
plt.ylabel('MEDV')

# 目标值
y = np.array(houses_df['MEDV'])
houses_df = houses_df.drop(['MEDV'], axis=1)

# 特征值
x = np.array(houses_df)

#数据集划分
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#数据处理
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#线性回归
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# 使用训练数据进行参数估计
lr.fit(x_train, y_train)

# 使用测试数据进行回归预测
y_test_pred = lr.predict(x_test)
#print(y_test_pred)
#print(y_test)

# 使用r2_score对模型评估
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 绘制预测值与真实值图
figure('预测值与真实值图模型的' + r'$R^2=%.4f$' % (r2_score(y_test_pred, y_test)), [y_test_pred, '预测值'], [y_test, '真实值'])

#平均绝对误差和均方误差
print('测试结果的平均绝对误差为:\n %s ' % mean_absolute_error(y_test_pred, y_test))
print('测试结果的均方误差为:\n %s ' % mean_squared_error(y_test_pred, y_test))

# 线性回归的系数
print('线性回归的系数为:\n w = %s \n b = %s' % (lr.coef_, lr.intercept_))


