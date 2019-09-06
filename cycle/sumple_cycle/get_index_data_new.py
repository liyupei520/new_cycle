#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cx_Oracle
import math
from scipy.fftpack import fft,ifft
from scipy import signal
import tushare as ts
from scipy import stats
import time


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
    conn = cx_Oracle.connect('reader/reader@172.16.50.232:1521/dfcf')
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




"""
从数据库获取信息
return  DataFrame  {SECURITYCODE,TRADEDATE,CLOSE}

"""        
def get_index_data(index_num=""):
    ##'000016',,'000300','000905'
    # and a.Tradedate >  to_date('2006-01-01 00:00:00','yyyy-mm-dd hh24:mi:ss')
    # and a.Tradedate >  to_date('1994-12-01 00:00:00','yyyy-mm-dd hh24:mi:ss')
    SQL = '''
            select  a.SECURITYCODE,a.TRADEDATE,a.NEW
            from DFCF.INDEX_TD_DAILY a  
            where a.SECURITYCODE = '{}'and a.Tradedate >  to_date('1990-06-01 00:00:00','yyyy-mm-dd hh24:mi:ss')
            
            '''
    SQL1 = """
            select  a.SECURITYCODE,a.TRADEDATE,a.NEW
            from DFCF.TRAD_SK_DAILY_JC a  
            where a.SECURITYCODE = '{}'and a.Tradedate >  to_date('1990-06-01 00:00:00','yyyy-mm-dd hh24:mi:ss')

    """
    #默认获取上证指数
    item="000001"
    if index_num:
        item=index_num
    sql = SQL.format(item)
    data=sql_select(sql)
    if not data:
        data = sql_select(SQL1.format(item))
    data_df=pd.DataFrame(data)
    data_df.columns=["SECURITYCODE","TRADEDATE","CLOSE"]
    return data_df


#生成同步序列
def get_data_sy(df,step):
    # 指数对数净值是以获取的第一个的上证指数价格为 1 元，然后取对数，取对数是为保证能够更容易看清指数走势
    df_1 = pd.Series(df[0], index=df.index, dtype='float')
    df_ln = np.log(df / df_1)

    #同比步长，12个月
    df_2 = df.shift(step)
    df_sy = np.log(df / df_2)
    df_all = pd.concat([df, df_ln, df_sy], axis=1)
    df_all.columns = [1, 2, 3]
    ax = df_all.plot(use_index=True, y=[1,  3], secondary_y=[ 3], figsize=(12, 9), title="上证指数，同步序列")
    ax.set_ylabel("指数")
    ax.right_ax.set_ylabel("上证指数同步序列")
    plt.grid(True)
    #plt.show()

    return [df, df_ln, df_sy]

def get_data_file():
    # 获取原始数据
    df_ = pd.read_excel("bit.xls")

    return df_

#数据清洗,生成同比序列
def get_clean_data_month(index_num=""):

    df=get_index_data(index_num) #从数据库获取原始数据
    # 将天换成月
    df['TRADEDATE']=df['TRADEDATE'].map(lambda x: x.strftime('%Y-%m'))
    #获取月最后一天收盘价作为这个月的收盘价
    df_1=df.drop_duplicates('TRADEDATE','last')
    df_ = pd.Series(df_1["CLOSE"],dtype='float')
    df_.index=df_1['TRADEDATE']
    #获取月平均做为数据
    # df_=df['CLOSE'].groupby(df['TRADEDATE']).mean()

    # df= get_data_file()[::-1]  # 从文件获取数据
    # df_1=df.drop_duplicates('date','last')
    # df_ = pd.Series(df_1["close"],dtype='float')
    # df_.index=df_1['date']
    print(df_)

    return get_data_sy(df_, 12)

#傅里叶变换，和频谱图
def get_fft(df_sy):

    # 频率
    # freqs_x=[n/(2*Ts*N) for n in range(N)]
    freqs = np.fft.fftfreq(len(df_sy), 1/12)
    y=df_sy
    yy = np.fft.fft(y)     #快速傅里叶变换
    yf = abs(yy)   #幅度
    plt.figure(dpi=100)
    plt.subplot(121)
    y.plot()
    plt.title('上证指数同比序列')

    plt.subplot(122)
    plt.plot(freqs, yf, 'r')
    #print("ddddd", sorted(yf))
    plt.title('傅里叶变换后频谱图')
    plt.xlabel('频率(年-1)')
    plt.ylabel('幅度')
    plt.grid(True)

    return yy,freqs


#简单拟合，按照42,100月
def get_cur(df_sy):
    if type(df_sy) is np.ndarray:
        df_sy=pd.Series(df_sy,range(len(df_sy)))
    else:
        df_sy.index=range(len(df_sy))
    # 拟合曲线
    from scipy.optimize import curve_fit

    # 创建函数模型用来生成数据
    def func(x, a, b, c, d, e):
        # fu=a*np.cos(np.pi*0.00111622*x+d) +b*np.cos(np.pi*0.00212637*x+e) - c
        fu = a * np.cos(np.pi * (2/ 42) * x + d) + b * np.cos(np.pi * (2 / 100) * x + e) + c
        return fu

    # 待拟合数据
    x = df_sy.index
    y = df_sy
    print("xxxx",x)
    print("yyyyy",y)
    # 使用curve_fit函数拟合噪声数据
    popt, pcov = curve_fit(func, x, y)
    print("popt",popt)
    # 输出给定函数模型func的最优参数
    #print(popt)
    df_func = []
    for i in df_sy.index:
        df_func.append(func(i, popt[0], popt[1], popt[2], popt[3], popt[4]))
    print(df_func)
    plt.figure(figsize=(12, 9))
    plt.title('模拟情况')
    plt.xlabel('iteration times')
    plt.ylabel('sy_rate')
    plt.plot(df_sy.index, df_func, color='green', label='new')
    plt.plot(df_sy.index, df_sy, color='red', label='original')
    plt.legend()  # 显示图例

#时间戳转时间
def timeStamp(timeNum):
    timeStamp = float(timeNum/1000)
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return  otherStyleTime

if __name__ == '__main__':
    #801192

    #国债 000012 企债 000013  上证指数：000001 深圳成指399001
    #data_list=get_clean_data_month('000012')
    data_list = get_clean_data_month('000001')
    #data_list_1 = get_clean_data_month('000001')
    df_sy=data_list[2]
    #df_sy_1 = data_list_1[2]
    # 去除为nan的项
    df_sy = df_sy.dropna()
    #df_sy_1 =df_sy_1.dropna()
    df_sy_old=df_sy.copy()

    #合并：
   # df_all=pd.concat([df_sy,df_sy_1],axis=1)
    #print(df_all)
    #相关系数
    #print(df_all.corr())

    #简单模拟曲线
    print("dfffff",df_sy)

    get_cur(df_sy)
    #get_cur(df_sy_1)
    #傅里叶变换
    yy,freqs=get_fft(df_sy)

    freqs_pf=pd.Series(yy,freqs)

    if 1/42 in freqs:
        print(111111111111111111111111111)
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
    #print("dfdffdf",df)
    f1=u
    gpl = np.exp(- ((fr - f1) / (2 * df)) ** 2)  # pos. frequencies
    gmn = np.exp(- ((fr + f1) / (2 * df)) ** 2)   # neg. frequencies
    # gpl = np.exp(-(fr - f1) ** 2 / (2 * df ** 2))
    # gmn = np.exp(-(fr + f1) ** 2 / (2 * df ** 2))
    #print("gpl,gmn", gpl, gmn)
    plt.figure(figsize=(12, 8))
    plt.plot(fr, gmn, 'r')
    plt.plot(fr, gpl)
    plt.title('filter (red)  + filtered spectrum')
    #plt.show()
    #print(x)
    gaussian_mat=gpl+gmn
    return gaussian_mat

#gau_data=imgGaussian_new(1/181,1/len(df_sy),freqs)


def get_t_yy(yy,T):
    t_freqs=1/T
    freqs_index_list=freqs_pf.index
    flag=2
    freqs_flag=0
    print("freqs_index_list",freqs_index_list)
    for item in freqs_index_list:
        if abs(abs(item)-t_freqs)<flag:
            flag=abs(abs(item)-t_freqs)
            freqs_flag=item
            #break

    yy_flag=pd.Series(freqs_pf[freqs_flag],range(len(yy)))
    print("yy_flfffff",flag,yy_flag)
    return  yy_flag


#傅里叶反变换
def iff_get(yy,T,freqs):
    #进行高斯过滤

    gau_data=imgGaussian_new(1 / T, 12/( len(yy)), freqs)



    yy_flag=yy
    #yy_flag=get_t_yy(yy,T)
    print("yy_flag",yy_flag)
    df_by_gau = []
    for i in range(len(yy_flag)):
        yy_flag_item=yy_flag[i]
        # if i>len(yy_flag)/2:
        #     yy_flag_item=yy_flag_item.conjugate()#求共轭复数
        df_by_gau.append( yy_flag_item* gau_data[i])
    print("df_by_gau",df_by_gau)
    plt.figure(figsize=(12, 9))
    plt.title('高斯滤波后,幅值')
    plt.xlabel('iteration times')
    plt.ylabel('sy_rate')
    plt.plot(freqs, [abs(item) for item in df_by_gau])

    #plt.legend()  # 显示图例
    #傅里叶反变换
    ifft_by_gau = ifft(df_by_gau)

    return df_by_gau,ifft_by_gau

def get_R(X,Y):
    w=np.array(X)
    b=np.array(Y)
    SSR = np.sum(np.multiply(w - b.mean(), w - b.mean()))
    SST = np.sum(np.multiply(b - b.mean(), b - b.mean()))
    R = SSR / SST
    return  R

"""
参数：
"""
#三种周期

t_0=42
t_1=100
t_2=200



df_by_gau,iyf_by_gau=iff_get(yy,t_0/12,freqs)
df_by_gau_1,iyf_by_gau_1=iff_get(yy,t_1/12,freqs)
df_by_gau_2,iyf_by_gau_2=iff_get(yy,t_2/12,freqs)

print(df_by_gau)
# plt.figure(figsize=(12, 9))
# plt.plot(freqs, df_by_gau)
# plt.title("高斯函数滤波")
# plt.grid(True)
#plt.show()

#模拟滤波曲线
get_cur(iyf_by_gau.real)

#三周期滤波后
plt.figure(figsize=(12, 9))
plt.title('高斯滤波后')
plt.xlabel('iteration times')
plt.ylabel('sy_rate')
#plt.plot(range(0,len(iyf_by_gau)), iyf_by_gau.real,  label="周期%s月"%t_0)
print("old",df_sy_old.index)
print("df_old",df_sy_old)
plt.plot(pd.to_datetime(df_sy_old.index), iyf_by_gau.real,  label="周期%s月"%t_0)
plt.plot(pd.to_datetime(df_sy_old.index), iyf_by_gau_1.real, label="周期%s月"%t_1)
plt.plot(pd.to_datetime(df_sy_old.index), iyf_by_gau_2.real,  label="周期%s月"%t_2)
plt.plot(pd.to_datetime(df_sy_old.index), df_sy, color='red', label='original')
plt.legend()  # 显示图例
#plt.show()

plt.figure(figsize=(12, 9))
plt.title('高斯滤波后，三周期合成')
plt.xlabel('iteration times')
plt.ylabel('sy_rate')
plt.plot(range(0,len(iyf_by_gau_2)), iyf_by_gau.real+iyf_by_gau_1.real+iyf_by_gau_2.real,  label="三周期相加合成")
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
    print("rrrrrrrrrr",get_R(y_hat,Y))
    print(y_hat,Y)
    print ("ppppppppp",stats.ttest_ind(y_hat, Y))
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
plt.plot(range(0,len(iyf_by_gau_2)), y_hat,  label="三周期线性回归合成")
plt.plot(df_sy.index, df_sy, color='red', label='original')
plt.legend()  # 显示图例
plt.show()