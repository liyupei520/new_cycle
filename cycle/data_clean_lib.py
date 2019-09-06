#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fftpack import fft,ifft
from scipy.optimize import curve_fit
from sklearn import linear_model

#设置中文显示
font = {'family' : 'SimHei',
    'weight' : 'bold',
    'size'  : '12'}
plt.rc('font', **font) # pass in the font dict as kwargs
plt.rc('axes',unicode_minus=False)


class Pd_Info:
    
    #生成同步序列
    def get_data_sy(self,df,step):
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

    def get_data_file(self,file_name):
        # 获取原始数据
        df_ = pd.read_excel("%s.xls"%file_name)

        return df_
    
    """
    主要函数：直线拟合--去除趋势；
    PCA 主成分分析法，降维
    """
    #直线拟合
    def get_straight_cur(self,df_sy):
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

    
    
    #去趋势，减去拟合的直线
    def get_avg_std(self,df):
        #去趋势
        df_cur = self.get_straight_cur(df.dropna()) #直线拟合
        df_cur.name=df.name
        return df_cur

    
    
    
    #数据清洗
    def data_day2month(self,file_name):
        df=self.get_data_file(file_name)
        df=df.dropna() #去除空行
        # 将天换成月
        if date in df.columns.tolist():
            df['date']=df['date'].map(lambda x: x.strftime('%Y-%m'))
            df_=df.drop_duplicates('date','last')
            df_.index= df_['date']
            df_=df_.drop("date",axis=1)
            return df_
        else:
            print("数据  日期列名需要为date ！！")
        
    
    
    


