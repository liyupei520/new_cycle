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



#设置中文显示
font = {'family' : 'SimHei',
    'weight' : 'bold',
    'size'  : '12'}
plt.rc('font', **font) # pass in the font dict as kwargs
plt.rc('axes',unicode_minus=False)

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
            where a.SECURITYCODE = '{}'and a.Tradedate >  to_date('1993-01-01 00:00:00','yyyy-mm-dd hh24:mi:ss')
            
            '''
    SQL1 = """
            select  a.SECURITYCODE,a.TRADEDATE,a.NEW
            from DFCF.TRAD_SK_DAILY_JC a  
            where a.SECURITYCODE = '{}'and a.Tradedate >  to_date('1993-12-01 00:00:00','yyyy-mm-dd hh24:mi:ss')

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
    #CRB取对数同比
    #同比步长，12个月
    df_2 = df.shift(step)
    df_sy = np.log(df / df_2)
    df_ln =np.log(df_2)
    df_sy=df_sy/df_ln
    df_all = pd.concat([df, df_ln, df_sy], axis=1)
    df_all.columns = [1, 2, 3]
    # ax = df_all.plot(use_index=True, y=[1,  3], secondary_y=[ 3], figsize=(12, 9), title="CRB对数同步序列")
    # ax.set_ylabel("date")
    # ax.right_ax.set_ylabel("同比")
    # plt.grid(True)
    #plt.show()

    return [df, df_ln, df_sy*100]

def get_data_file(file_name):
    # 获取原始数据
    df_ = pd.read_excel("%s.xls"%file_name)

    return df_

#数据清洗
def get_clean_data_month(file_name,*argas):
    df=get_data_file(file_name)
    df=df.dropna() #去除空行
    # 将天换成月
    df['date']=df['date'].map(lambda x: x.strftime('%Y-%m'))
    df_=df.drop_duplicates('date','last')
    df_.index= df_['date']
    df_=df_.drop("date",axis=1)
    return df_

#傅里叶变换，和频谱图
def get_fft(df_sy):

    # 采样频率(按年)
    f_s = 12
    freqs = np.fft.fftfreq(len(df_sy),1/f_s) #采样间距（采样率的倒数）
    y=df_sy
    yy = np.fft.fft(y)     #快速傅里叶变换
    yf = abs(yy)   #幅度

    plt.figure(figsize=(12,10))
    plt.subplot(131)
    ax=y.plot(title='%s原始序列'%df_sy.name)
    ax.set_xlabel(u"r日期")
    xticks = range(0, len(y), len(y)//6)
    xticklabels = [y.index[i] for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=70)  # roration 旋转

    plt.subplot(132)
    plt.plot(freqs, yf, 'r')
    plt.title('%s傅里叶变换后频谱图'%df_sy.name)
    plt.xlabel('频率 年（-1）')
    plt.ylabel('幅度')
    plt.grid(True)
    
    print("0000",freqs)
    print("1111",12/freqs)
    print("2222",yf)
    print(len(yf))
    # pf=pd.Series(yf,12/freqs)
    # pf=pf
    half=len(yf)//2
    print(half)
    plt.subplot(133)
    plt.plot(12/freqs[:half], yf[:half], 'r')
    plt.title('%s傅里叶变换后频谱图'%df_sy.name)
    plt.xlabel('周期 （月）')
    plt.ylabel('幅度')
    plt.grid(True)

    return yy,freqs


#简单拟合，按照42,100月
def get_cur(df_sy):
    df_old=df_sy.copy()
    df_sy.index=range(len(df_sy))
    # 拟合曲线
    from scipy.optimize import curve_fit

    # 创建函数模型用来生成数据
    def func(x, a, b, c, d, e):
        # fu=a*np.cos(np.pi*0.00111622*x+d) +b*np.cos(np.pi*0.00212637*x+e) - c
        fu = a * np.cos(np.pi * (2/ 40) * x + d) + b * np.cos(np.pi * (2 / 66) * x + e) - c
        return fu

    # 待拟合数据
    x = df_sy.index
    y = df_sy
    #print("xxxx",x)
    #print("yyyyy",y)
    # 使用curve_fit函数拟合噪声数据
    print("xxxxxx",x)
    print("yyyyy",y)
    popt, pcov = curve_fit(func, x, y)
    #print("popt",popt)
    # 输出给定函数模型func的最优参数
    #print(popt)
    df_func = []
    for i in df_sy.index:
        df_func.append(func(i, popt[0], popt[1], popt[2], popt[3], popt[4]))

    plt.figure(figsize=(12,5))
    plt.title('模拟情况')
    plt.xlabel('iteration times')
    plt.ylabel('sy_rate')
    plt.plot(df_old.index, df_func, color='green', label='new')
    plt.plot(df_old.index, df_sy, color='red', label='original')
    plt.legend()  # 显示图例



"""
主要函数：直线拟合--去除趋势；
        PCA 主成分分析法，降维
"""
#直线拟合
def get_straight_cur(df_sy):
    df_old= df_sy.copy()
    if type(df_sy) is np.ndarray:
        df_sy=pd.Series(df_sy,range(len(df_sy)))
    else:
        df_sy.index=range(len(df_sy))
    # 拟合直线
    from scipy.optimize import curve_fit

    # 创建函数模型用来生成数据
    def func(x, a, b):
        fu = a * x + b
        return fu

    # 待拟合数据
    x = df_sy.index
    y = df_sy
    # 使用curve_fit函数拟合噪声数据
    popt, pcov = curve_fit(func, x, y)
    # 输出给定函数模型func的最优参数
    #print(popt)
    df_func = []
    for i in df_sy.index:
        df_func.append(func(i, popt[0], popt[1]))
    df_cur=pd.Series(df_func,df_old.index)
    df_new=df_old-df_cur

    df_all=pd.concat([df_new,df_old,df_cur],axis=1)
    df_all.columns=["new","old","fit"]
    ax = df_all.plot(use_index=True, figsize=(12, 5), title=u"趋势情况")
    ax.set_ylabel(u"%")
    ax.set_xlabel(u"r日期")
    xticks = range(0, len(df_all), 6)
    xticklabels = [df_all.index[i] for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=70)  # roration 旋转

    return df_new


# 3个月平均去噪
def three_m(df,*args):
    list_A = args
    df_1 = df.copy()
    for item in list_A:
        for i in  range(2,len(df_1)-3):
            df_1[item][i]=df[item][i-2:i+3].mean()  #5个月做平滑处理

    return df_1

#3个月平滑，去趋势，标准化
def m_avg_std(df,*args):
    list_A=args
    df_1=df.copy() #复制
    df = three_m(df, *args)  # 3个月平滑
    #去趋势，标准化
    for item in list_A:
        df_cur = get_straight_cur(df[item]) #直线拟合
        std_max = 10
        df_max = df_cur.max()
        df_min=df_cur.min()
        df_max_min=df_max-df_min
        df_1[item] = df_cur * std_max / df_max_min

    return df_1

#标准化
def get_std(df):
    df_1=df.copy()
    #去趋势，标准化
    for item in df.columns:
        df_cur = df[item]
        std_max = 10
        df_max = df_cur.max()
        df_min=df_cur.min()
        df_max_min=df_max-df_min
        df_1[item] = df_cur * std_max / df_max_min

    return df_1


#去趋势，减去拟合的直线
def get_avg_std(df):
    #去趋势
    df_cur = get_straight_cur(df.dropna()) #直线拟合
    df_cur.name=df.name
    return df_cur

"""
PCA方法
对数据进行归一化处理（代码中并不是这么做的，而是直接减去均值）
计算归一化后的数据集的协方差矩阵
计算协方差矩阵的特征值和特征向量
保留最重要的k个特征（通常k要小于n）。也能够自己制定。也能够选择一个阈值，然后通过前k个特征值之和减去后面n-k个特征值之和大于这个阈值，则选择这个k
找出k个特征值相应的特征向量
将m * n的数据集乘以k个n维的特征向量的特征向量（n * k）,得到最后降维的数据。

"""
#计算均值,要求输入数据为numpy的矩阵格式，行表示样本数，列表示特征
def meanX(dataX):
    return np.mean(dataX,axis=0)#axis=0表示依照列来求均值。假设输入list,则axis=1
"""
參数：
    - XMat：传入的是一个numpy的矩阵格式，行表示样本数，列表示特征    
    - k：表示取前k个特征值相应的特征向量
返回值：
    - finalData：參数一指的是返回的低维矩阵，相应于输入參数二
    - reconData：參数二相应的是移动坐标轴后的矩阵
"""
def pca(XMat, k):
    data_adjust = np.mat(XMat).T  #创建矩阵，并转置
    # 求均值
    mean_matrix = np.mean(data_adjust, axis=1)
    data_adjust = data_adjust - mean_matrix
    n = len(data_adjust) #原始维度
    covX = np.cov(data_adjust,rowvar=True)   #计算协方差矩阵 rowvar:默认为True,此时每一行代表一个变量（属性），每一列代表一个观测；为False时，则反之

    featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量
    # print("covX", covX,featValue,featVec)
    index = np.argsort(-featValue) #依照featValue进行从大到小排序
    finalData = []
    if k > n:
        print ("k must lower than feature number")
        return
    else:
        #注意特征向量是列向量。而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = featVec.T[index[:k]] #所以这里须要进行转置 ,按照排列取特征向量
        finalData = selectVec*data_adjust #提取主成分
        finalData=pd.DataFrame(finalData.T,XMat.index,columns=["PCA"])
    return finalData

#获取经济周期指标
def get_cp_pp():
    # cpi_ppi.xls  ,ten_year.xls
    df_cp_pp = get_clean_data_month('cpi_ppi')
    df_cp_pp=df_cp_pp.shift(1) #延迟一期
    df_ten = get_clean_data_month('ten_year')
    df_all = pd.concat([df_ten, df_cp_pp], axis=1, sort=True).dropna()
    # 获取对数同比序列，12个月
    df_all["CRB"] = get_data_sy(df_all["CRB"], 12)[2]
    df_all=df_all.dropna()
    df_all_old=df_all.copy()
    df_all = m_avg_std(df_all, "CRB", "CPI", "PPI", "ten_year") #平滑，去趋势，标准化
    pca_df = pca(df_all[["CRB", "CPI", "PPI", "ten_year"]].dropna(), 1)  # 主成分分析
    #pca_df = three_m(pca_df, "PCA")  # 平滑处理

    df_all = pd.concat([df_all, pca_df], axis=1, sort=True).dropna()

    ax = df_all.plot(use_index=True, y=["CRB", "CPI", "PPI", "ten_year"]
                   , figsize=(10, 6), title=u"经济周期指标")
    pca_df["PCA"].plot(linewidth='5', label="合成")
    ax.set_ylabel(u"比例")
    # ax.right_ax.set_ylabel(u"CRB")
    plt.grid(True)
    xticks=range(1, len(df_all) + 1, 6)
    xticklabels=[df_all.index[i] for i in xticks ]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=90) #roration 旋转

    return df_all_old


#获取流动性指标
def get_MM():
    # cpi_ppi.xls  ,ten_year.xls
    df_mm = get_clean_data_month('m_12')
    df_mm=df_mm.shift(1)  #延迟一期
    df_one = get_clean_data_month('one_year')

    df_all = pd.concat([df_one, df_mm], axis=1, sort=True)
    df_all= df_all[["one_year","M1_t","M2_t","M1_M2_t2"]].dropna()

    df_all = m_avg_std(df_all, "one_year","M1_t","M2_t","M1_M2_t2") #平滑，去趋势，标准化
    print("mm_df_all",df_all)
    pca_df = pca(df_all[["M1_t","M2_t","one_year","M1_M2_t2"]].dropna(), 1)  # 主成分分析，只取第一个
    #pca_df=-pca_df
    print("pca_df", pca_df)
    #pca_df = three_m(pca_df, "PCA") #平滑处理
    #减去cp_pp周期
    pca_df_1 = get_cp_pp()
    print("pca_df",pca_df)
    print("pca_df_1",pca_df_1)
    pca_df_2=pca_df + pca_df_1


    df_all = pd.concat([df_all, pca_df], axis=1, sort=False).dropna()
    ax_1 = df_all.plot(use_index=True, y=["M1_t","M2_t","one_year","M1_M2_t2"]
                     , figsize=(10, 6), title=u"流动性周期指标")
    #ax_1.plot(pca_df, linewidth='3', label="合成")
    ax_1.plot(pca_df_2, linewidth='3', label="合成")
    #ax_1.plot(pca_df_1,linewidth='3', label="合成")
    ax_1.set_ylabel(u"比例")
    # ax.right_ax.set_ylabel(u"CRB")
    ax_1.grid(True)
    xticks=range(1, len(df_all) + 1, 6)
    xticklabels=[df_all.index[i] for i in xticks ]
    ax_1.set_xticks(xticks)
    ax_1.set_xticklabels(xticklabels, rotation=90) #roration 旋转

    return df_all


#高斯滤波器
def imgGaussian_new(u,sigma,freqs):
    '''
    :param sigma: σ标准差
    :return: 高斯滤波后
    '''
    #u = 0  # 均值μ
    fr=freqs
    df=sigma
    f1=u
    gpl = np.exp(- ((fr - f1) / (2 * df)) ** 2)
    gmn = np.exp(- ((fr + f1) / (2 * df)) ** 2)  
    # gpl = np.exp(-(fr - f1) ** 2 / (2 * df ** 2))
    # gmn = np.exp(-(fr + f1) ** 2 / (2 * df ** 2))
    gaussian_mat=gpl+gmn
    return gaussian_mat



#傅里叶反变换
def iff_get(yy,T,freqs,df_sy):
    #进行高斯过滤
    gau_data=imgGaussian_new(12 / T, 12/( len(yy)), freqs)

    yy_flag=yy
    #yy_flag=get_t_yy(yy,T)

    df_by_gau = []
    for i in range(len(yy_flag)):
        yy_flag_item=yy_flag[i]
        # if i>len(yy_flag)/2:
        #     yy_flag_item=yy_flag_item.conjugate()#求共轭复数
        df_by_gau.append( yy_flag_item* gau_data[i])

    #傅里叶反变换
    ifft_by_gau = ifft(df_by_gau)
    ifft_df=pd.Series(ifft_by_gau.real,df_sy.index)
    ifft_df.name =T
    return ifft_df


#拟合曲线
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
    #print(X)
    #训练模型
    clf.fit(X,Y)

    # 计算y_hat
    y_hat = clf.predict(X)
    return y_hat



#三周期滤波，拟合合成
def three_cycle_fit(df_sy):

    # 傅里叶变换
    yy, freqs = get_fft(df_sy)  #快速傅里叶变换，频率

    #三种周期
    t_0=42
    t_1=100
    t_2=200

    #傅里叶反变换
    iyf_by_gau=iff_get(yy,t_0,freqs,df_sy)
    iyf_by_gau_1=iff_get(yy,t_1,freqs,df_sy)
    iyf_by_gau_2=iff_get(yy,t_2,freqs,df_sy)

    # #三周期滤波后，画图
    df_all = pd.concat([df_sy, iyf_by_gau,iyf_by_gau_1,iyf_by_gau_2], axis=1, sort=False)
    ax_1 = df_all.plot(use_index=True
                       , figsize=(12, 6), title=u"%s高斯滤波后"%df_sy.name)
    ax_1.set_xlabel(u"日期")
    ax_1.set_ylabel(u"单位（%）")
    ax_1.grid(True)
    xticks = range(0, len(df_all), 6)
    xticklabels = [df_all.index[i] for i in xticks]
    ax_1.set_xticks(xticks)
    ax_1.set_xticklabels(xticklabels, rotation=70)  # roration 旋转

    #三周期拟合，线性回归值
    y_hat=get_fit(iyf_by_gau.real,iyf_by_gau_1.real,iyf_by_gau_2.real,df_sy)

    #画图
    three_fit_df=pd.Series(y_hat,df_sy.index)
    three_fit_df.name="%s_fit"%df_sy.name
    global df_ch_all
    df_ch_all.append(three_fit_df)
    df_all_fit = pd.concat([df_sy,three_fit_df ], axis=1, sort=False)
    ax_2 = df_all_fit.plot(use_index=True
                       , figsize=(12,6), title=u"%s高斯滤波后，三周期线性回归合成"%df_sy.name)
    ax_2.set_xlabel(u"日期")
    ax_2.set_ylabel(u"单位（%）")
    ax_2.grid(True)
    xticks_2 = range(0, len(df_all_fit), 6)
    xticklabels_2 = [df_all_fit.index[i] for i in xticks_2]
    ax_2.set_xticks(xticks_2)
    ax_2.set_xticklabels(xticklabels_2, rotation=70)  # roration 旋转


def show_from_file(file_path):
    df_from_file = get_clean_data_month(file_path)
    for item in df_from_file.columns:
        df_ = df_from_file[item][90:]  #取每一列数据
        df_= get_avg_std(df_)  #去趋势
        three_cycle_fit(df_)  #拟合

def show_from_df(df_all):
    df_all_fit=pd.concat(df_all,axis=1)
    df_all_fit=get_std(df_all_fit.dropna()) #标准化

    pca_df = pca(df_all_fit, 1) #主成分分析
    ax_2 = df_all_fit.plot(use_index=True
                           , figsize=(12, 6), title=u"三周期线性回归合成集合")
    ax_2.set_xlabel(u"日期")
    ax_2.set_ylabel(u"单位（%）")
    pca_df["PCA"].plot(linewidth='5', label="合成")
    ax_2.grid(True)
    xticks_2 = range(0, len(df_all_fit), 6)
    xticklabels_2 = [df_all_fit.index[i] for i in xticks_2]
    ax_2.set_xticks(xticks_2)
    ax_2.set_xticklabels(xticklabels_2, rotation=70)  # roration 旋转

if __name__ == '__main__':
    df_ch_all=[]
    #获取数据
    df_cp_pp = get_clean_data_month('PPI_CPI_month')
    df_cpi = df_cp_pp["CPI"][15:]  # 1998-01开始
    df_ppi=df_cp_pp["PPI"]
    df_cpi = get_avg_std(df_cpi)  #去趋势
    df_ppi = get_avg_std(df_ppi)  #去趋势
    three_cycle_fit(df_cpi)
    #three_cycle_fit(df_ppi)     #三周期拟合

    # #获取数据
    # df_crb = get_clean_data_month('CRB_index')
    # df_crb = df_crb["CRB"][600:]
    # df_crb=get_data_sy(df_crb, 12)[2].dropna()
    # df_crb = get_avg_std(df_crb)
    # three_cycle_fit(df_crb)


    # df_M1M2 = get_clean_data_month('M1_M2')
    # df_M1 = df_M1M2["M1"]
    # df_M1= get_avg_std(df_M1)
    # three_cycle_fit(df_M1)

    # df_M2 = df_M1M2["M2"]
    # df_M2= get_avg_std(df_M2)
    # three_cycle_fit(df_M2)
    # #
    # df_M1M2["M1-M2"] = df_M1M2["M1"]-df_M1M2["M2"]
    # df_M1_M2=df_M1M2["M1-M2"]
    # df_M1_M2= get_avg_std(df_M1_M2)
    # three_cycle_fit(df_M1_M2)
    # #
    # df_ten_one = get_clean_data_month('ten_year_one_year')
    # df_ten = df_ten_one["ten_year"]
    # df_ten= get_avg_std(df_ten)
    # three_cycle_fit(df_ten)

    # print(df_ch_all)

    # show_from_file("house_start")
    # show_from_df(df_ch_all)
    plt.show()                  #画图