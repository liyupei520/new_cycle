#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cx_Oracle
from scipy import signal
import tushare as ts
#设置中文显示
font = {'family' : 'SimHei',
    'weight' : 'bold',
    'size'  : '12'}
plt.rc('font', **font) # pass in the font dict as kwargs
plt.rc('axes',unicode_minus=False)

df_sh=ts.get_hist_data('000001',start='1996-01-05',end='2019-05-15')
print(len(df_sh))
print(df_sh)
data=df_sh
data_se=pd.DataFrame(data)
df_ = pd.Series(data_se[2], dtype='float')
#df_.index = data_se[1]
#上证指数对数净值是以获取的第一个的上证指数价格为 1 元，然后取对数，取对数是为保证能够更容易看清指数走势
df_1=pd.Series(df_[0],index=df_.index,dtype='float')
df_ln=np.log(df_/df_1)

#上证指数同比序列
#按照周期240天，与240天的价格相除，同时取对数
df_2=df_.shift(240)
df_sy=np.log(df_/df_2)
#print(df_ln)
df_all=pd.concat([df_,df_ln,df_sy],axis=1)
df_all.columns = [1, 2,3]
ax = df_all.plot(use_index=True, y=[1, 2,3], secondary_y=[2,3], figsize=(12, 9), title="上证指数及对数净值及同步序列")
ax.set_ylabel("指数")
ax.right_ax.set_ylabel("上证指数对数净值")
#plt.show()
print(df_)
#带通滤波,同步序列上进行滤波
b, a = signal.butter(8, [0.00057, 0.00213], 'bandpass')
print("this is b,a:",b,a)
df_sy=df_sy.dropna()
filtedData = signal.filtfilt(b, a, df_sy)
print(filtedData,"len",len(filtedData))
df_all=pd.DataFrame(df_sy)
df_all["2"]=filtedData
df_all.columns = [1, 2]
print(df_all)
ax = df_all.plot(use_index=True, y=[1, 2], secondary_y=[2], figsize=(12, 9), title="上证指数及同步序列滤波")
ax.set_ylabel("指数")
ax.right_ax.set_ylabel("上证指数同步序列滤波")
#plt.show()


    #傅里叶变换
    # print(df_sy)
    # df_f =np.fft.fft(df_sy)
    # print(df_f)
    # df_f.plot( figsize=(12, 9),title="上证指数同步序列傅里叶变换")
    # plt.show()

import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt

#采样点个数：3489，len(df_sy)

N=len(df_sy)
dt=1   #最小采样间隔为天
print(df_sy)
x=df_sy.index
print("xxxxx",x)
#设置需要采样的信号
y=df_sy
print("yyyyyyy",y)
yy=fft(y)                     #快速傅里叶变换
freqs = np.fft.fftfreq(len(yy))
print("222222222222222",freqs)
yf=abs(fft(y))                # 取绝对值
yf1=abs(fft(y))/len(x)           #归一化处理
yf2 = yf1[range(int(len(x)/2))]  #由于对称性，只取一半区间

xf = np.arange(len(y))        # 频率
xf1 = xf
xf2 = xf[range(int(len(x)/2))]  #取一半区间

plt.figure(dpi=100)
plt.subplot(211)
plt.plot(x,y)
plt.title('上证指数同比序列')

plt.subplot(212)
plt.plot(freqs,yf,'r')
plt.title('傅里叶变换后频谱图',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表


#过滤不明显的频率，
yf3=[]
for i in range(21):
    yf3.append(yy[i])
for i in range(21,len(yy)-21):
    yf3.append(0)
for i in range(len(yy)-21,len(yy)):
    yf3.append(yy[i])



#傅里叶反变换
iyf3=ifft(yf3)

plt.figure(figsize=(12, 9))
plt.title('去除噪声后')
plt.xlabel('iteration times')
plt.ylabel('sy_rate')
plt.plot(range(240,len(iyf3)+240), iyf3.real, color='green', label='new')
plt.plot(df_sy.index, df_sy, color='red', label='original')
plt.legend()  # 显示图例
#plt.show()


#只保留明显的频率,2,4,8
yf4=[]
for i in range(len(yf3)):
    if i>=2 and i<=8:
        yf4.append(yf3[i])

    else:
        yf4.append(0)

iyf4=ifft(yf4)

plt.figure(figsize=(12, 9))
plt.title('保留[2,8]')
plt.xlabel('iteration times')
plt.ylabel('sy_rate')
plt.plot(range(240,len(iyf4)+240), iyf4.real, color='green', label='new')
plt.plot(df_sy.index, df_sy, color='red', label='original')
plt.legend()  # 显示图例
#plt.show()

#拟合曲线
from scipy.optimize import curve_fit

#创建函数模型用来生成数据
def func(x, a, b,c,d,e):
        fu=a*np.cos(np.pi*0.00111622*x+d) +b*np.cos(np.pi*0.00212637*x+e) - c
        return fu

#待拟合数据
x = df_sy.index
y = df_sy

#使用curve_fit函数拟合噪声数据
popt, pcov = curve_fit(func, x, y)

#输出给定函数模型func的最优参数
print(popt)
df_func=[]
for i in df_sy.index:
    df_func.append(func(i, popt[0],popt[1], popt[2], popt[3], popt[4]))

plt.figure(figsize=(12, 9))
plt.title('模拟情况')
plt.xlabel('iteration times')
plt.ylabel('sy_rate')
plt.plot(df_sy.index, df_func, color='green', label='new')
plt.plot(df_sy.index, df_sy, color='red', label='original')
plt.legend()  # 显示图例
plt.show()
