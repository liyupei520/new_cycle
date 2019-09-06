#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cx_Oracle
import math
from scipy import signal
import tushare as ts
#设置中文显示
font = {'family' : 'SimHei',
    'weight' : 'bold',
    'size'  : '12'}
plt.rc('font', **font) # pass in the font dict as kwargs
plt.rc('axes',unicode_minus=False)

#写到csv
def write_csv(data,file_name):
    outputFilename = '%s.csv' %file_name
    # with open(outputFilename, 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['指数名称', '日期','收盘'])
    #     #writer.writerows(data)
    data.columns = ['Ticker', 'Date','Close']
    data.to_csv(outputFilename,encoding='utf-8')

def sql_select(sql):
    # 使用cursor()方法获取操作游标
    conn = cx_Oracle.connect('reader/reader@172.16.124.92:1521/dfcf')
    cursor = conn.cursor()
    # # 使用execute方法执行SQL语句
    cursor.execute(sql)
    # 使用fetchone()方法获取一条数据
    #data=cursor.fetchone()
    # 获取所有数据
    all_data = cursor.fetchall()
    # 获取部分数据，8条
    # many_data=cursor.fetchmany(8)
    cursor.close()
    conn.close()
    return  all_data

# and a.Tradedate >  to_date('2006-01-01 00:00:00','yyyy-mm-dd hh24:mi:ss')
SQL = '''
        select  a.SECURITYCODE,a.TRADEDATE,a.NEW
        from DFCF.INDEX_TD_DAILY a  
        where a.SECURITYCODE = '{}' 
        

        '''

##'000016',,'000300','000905'
indexs=['000001']
#获取上证指数
item="000001"
sql = SQL.format(item)
data=sql_select(sql)
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
df_all=pd.concat([df_,df_ln,df_sy],axis=1)
df_all.columns = [1, 2,3]
ax = df_all.plot(use_index=True, y=[1, 2,3], secondary_y=[2,3], figsize=(12, 9), title="上证指数及对数净值及同步序列")
ax.set_ylabel("指数")
ax.right_ax.set_ylabel("上证指数对数净值")
#plt.show()
print(df_)

# #带通滤波,同步序列上进行滤波
# b, a = signal.butter(8, [0.00057, 0.00213], 'bandpass')
# print("this is b,a:",b,a)
# df_sy=df_sy.dropna()
# filtedData = signal.filtfilt(b, a, df_sy)
# print(filtedData,"len",len(filtedData))
# df_all=pd.DataFrame(df_sy)
# df_all["2"]=filtedData
# df_all.columns = [1, 2]
# print(df_all)
# ax = df_all.plot(use_index=True, y=[1, 2], secondary_y=[2], figsize=(12, 9), title="上证指数及同步序列滤波")
# ax.set_ylabel("指数")
# ax.right_ax.set_ylabel("上证指数同步序列滤波")
#plt.show()


from scipy.fftpack import fft,ifft
df_sy_cut=df_sy[240:]
#去除为nan的项
df_sy_cut.index=range(len(df_sy_cut))
print(df_sy_cut)

df_sy=df_sy.dropna()
#采样点个数：6701，len(df_sy)
N=len(df_sy)
#采样频率，一年即240天
Fs =240
#采样间隔
Ts =1/Fs
print(df_sy)
x=df_sy.index
#频率
#freqs_x=[n/(2*Ts*N) for n in range(N)]
freqs = np.fft.fftfreq(len(df_sy),1)
print("xxxxx",x)
#设置需要采样的信号
y=df_sy
print("yyyyyyy",y)
yy=fft(y)                     #快速傅里叶变换

print("222222222222222",freqs)
yf=abs(fft(y))                # 取绝对值,幅度

plt.figure(dpi=100)
plt.subplot(211)
plt.plot(x,y)
plt.title('上证指数同比序列')

plt.subplot(212)
plt.plot(freqs,yf,'r')
print("ddddd",sorted(yf))
plt.title('傅里叶变换后频谱图',fontsize=7,color='#7A378B')

fft_df=yf

#过滤不明显的频率，
yf3=[]
for i in range(51):
    yf3.append(yy[i])
for i in range(51,len(yy)-51):
    yf3.append(0)
for i in range(len(yy)-51,len(yy)):
    yf3.append(yy[i])

#傅里叶反变换
iyf3=ifft(yf3)
print("yf3333333333",yf3)
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
    if i<15 or i>len(yf3)-i:
        yf4.append(yf3[i])
    else:
        yf4.append(0)
print("yf444444444444444444",yf4)
iyf4=ifft(yf4)

plt.figure(figsize=(12, 9))
plt.title('保留[15,-15]')
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
        #fu=a*np.cos(np.pi*0.00111622*x+d) +b*np.cos(np.pi*0.00212637*x+e) - c
        fu = a * np.cos(np.pi * (1/420) * x + d) + b * np.cos(np.pi * (1/1000) * x + e) - c
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
#plt.show()


# 高斯滤波器
def imgGaussian(u,sigma,freqs):
    '''
    :param sigma: σ标准差
    :return: 高斯滤波器的模板
    '''
    #u = 0  # 均值μ
    #u01 = -2
    #sig = math.sqrt(0.2)  # 标准差δ
    #标准差，3个标准差之内的占比99%
    #x = np.linspace(u - 3 * sigma, u + 3 * sigma, freqs)
    x=freqs
    print(x)
    y_sig = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    gaussian_mat=y_sig
    #归一化处理
    MAX_Y=1/ (math.sqrt(2 * math.pi) * sigma)
    print("gggg",gaussian_mat)
    gaussian_mat=gaussian_mat/MAX_Y
    print("aaaaa",MAX_Y,gaussian_mat)
    plt.figure(figsize=(12, 9))
    plt.plot(x, gaussian_mat)
    plt.title("正太分布")
    plt.grid(True)
    #plt.show()
    return gaussian_mat

def imgGaussian_new(u,sigma,freqs):
    '''
    :param sigma: σ标准差
    :return: 高斯滤波器的模板
    '''
    #u = 0  # 均值μ
    fr=freqs
    df=sigma
    f1=u
    gpl = np.exp(- ((fr - f1) / (2 * df)) ** 2)  # pos. frequencies
    gmn = np.exp(- ((fr + f1) / (2 * df)) ** 2)   # neg. frequencies
    print("gpl,gmn", gpl, gmn)
    plt.figure(figsize=(12, 8))
    plt.plot(fr, gmn, 'r')
    plt.plot(fr, gpl)
    plt.title('filter (red)  + filtered spectrum')
    #plt.show()
    print(x)
    gaussian_mat=gpl+gmn
    return gaussian_mat

#gau_data=imgGaussian_new(1/181,1/len(df_sy),freqs)

#傅里叶反变换
def iff_get(yy,T,freqs):
    #进行高斯过滤
    gau_data=imgGaussian_new(1 / T, 1/ len(yy)*2, freqs)

    df_by_gau = []
    for i in range(len(fft_df)):
        df_by_gau.append(yy[i] * gau_data[i])

    #傅里叶反变换
    ifft_by_gau = ifft(df_by_gau)

    return df_by_gau,ifft_by_gau


"""
参数：
"""
#三种周期

t_0=840
t_1=4000
t_2=2000

df_by_gau,iyf_by_gau=iff_get(yy,t_0,freqs)
df_by_gau_1,iyf_by_gau_1=iff_get(yy,t_1,freqs)
df_by_gau_2,iyf_by_gau_2=iff_get(yy,t_2,freqs)

print(df_by_gau)
plt.figure(figsize=(12, 9))
plt.plot(freqs, df_by_gau)
plt.title("高斯函数滤波")
plt.grid(True)
#plt.show()



#三周期滤波后
plt.figure(figsize=(12, 9))
plt.title('高斯滤波后')
plt.xlabel('iteration times')
plt.ylabel('sy_rate')
plt.plot(range(240,len(iyf_by_gau)+240), iyf_by_gau.real,  label="周期%s天"%t_0)
plt.plot(range(240,len(iyf_by_gau_1)+240), iyf_by_gau_1.real, label="周期%s天"%t_1)
plt.plot(range(240,len(iyf_by_gau_2)+240), iyf_by_gau_2.real,  label="周期%s天"%t_2)
plt.plot(df_sy.index, df_sy, color='red', label='original')
plt.legend()  # 显示图例
#plt.show()

plt.figure(figsize=(12, 9))
plt.title('高斯滤波后，三周期合成')
plt.xlabel('iteration times')
plt.ylabel('sy_rate')
plt.plot(range(240,len(iyf_by_gau_2)+240), iyf_by_gau.real+iyf_by_gau_1.real+iyf_by_gau_2.real,  label="三周期相加合成")
plt.plot(df_sy.index, df_sy, color='red', label='original')
plt.legend()  # 显示图例
#plt.show()


#xianx
from sklearn import linear_model
#拟合函数
def get_fit(x_0,x_1,x_2,df_sy):

    #调用模型
    clf = linear_model.LinearRegression()
    #待拟合数据
    X =[]
    for i in range(len(x_0)):
        X.append([x_0[i],x_1[i],x_2[i]])
    Y = df_sy
    print(X)
    #训练模型
    clf.fit(X,Y)
    # 计算y_hat
    y_hat = clf.predict(X)
    print(y_hat)
    print(clf.coef_) #系数
    print(clf.intercept_)  #截距
    return y_hat

#线性回归值
y_hat=get_fit(iyf_by_gau.real,iyf_by_gau_1.real,iyf_by_gau_2.real,df_sy)

plt.figure(figsize=(12, 9))
plt.title('高斯滤波后，三周期线性回归合成')
plt.xlabel('iteration times')
plt.ylabel('sy_rate')
plt.plot(range(240,len(iyf_by_gau_2)+240), y_hat,  label="三周期线性回归合成")
plt.plot(df_sy.index, df_sy, color='red', label='original')
plt.legend()  # 显示图例
plt.show()