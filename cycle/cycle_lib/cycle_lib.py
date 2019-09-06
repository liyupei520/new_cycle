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


class Cycle_PF:
    def __init__ (self, pd):
        self.df_sy = pd
    
    #傅里叶变换
    def __get_fft(self):
        df_sy=self.df_sy
        f_s = 12                                  #采样频率(按年)    
        freqs = np.fft.fftfreq(len(df_sy),1/f_s)  #采样间距（采样率的倒数）
        df_fft = np.fft.fft(df_sy)                #快速傅里叶变换
        
        return df_fft,freqs
        
    #高斯滤波器
    def __imgGaussian_new(self,u,sigma,freqs):
        gpl = np.exp(- ((freqs - u) / (2 * sigma)) ** 2)
        gmn = np.exp(- ((freqs + u) / (2 * sigma)) ** 2)  
        gaussian_mat=gpl+gmn
        
        return gaussian_mat



    #傅里叶反变换
    def __get_ifft(self,T,df_fft,freqs):
        
        #高斯过滤
        gau_data=self.__imgGaussian_new(12 / T, 12/( len(df_fft)), freqs)
        
        df_by_gau = []
        for i in range(len(df_fft)):
            yy_flag_item=df_fft[i]
            df_by_gau.append( yy_flag_item* gau_data[i])
        
        ifft_by_gau = ifft(df_by_gau) #傅里叶反变换
        
        ifft_df=pd.Series(ifft_by_gau.real,self.df_sy.index)
        ifft_df.name =T
        
        return ifft_df

    #傅里叶变换
    def show_fft(self):
        df_fft,freqs=self.__get_fft()
        df_fft_abs = abs(df_fft)            #幅度
        df_sy=self.df_sy
        
        plt.figure(figsize=(12,5))
        plt.subplot(121)
        ax=df_sy.plot(title='%s原始序列'%df_sy.name)
        ax.set_xlabel(u"日期")
        # xticks = range(0, len(df_sy), len(df_sy)//6)
        # xticklabels = [df_sy.index[i] for i in xticks]
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(xticklabels, rotation=70)  # roration 旋转

        plt.subplot(122)
        plt.plot(freqs, df_fft_abs, 'r')
        plt.title('%s傅里叶变换后频谱图'%df_sy.name)
        plt.xlabel('频率 年（-1）')
        plt.ylabel('幅度')
        plt.grid(True)
        plt.show()
        
    #周期滤波
    def cycle_filter(self,*args):
        # 傅里叶变换
        df_fft, freqs = self.__get_fft()  #快速傅里叶变换，频率
        filter_dfs=[]
        for item_t in args:
            filter_dfs.append(self.__get_ifft(item_t,df_fft,freqs))
        
        return filter_dfs
    
    def show_cycle_filter(self,*args):
        df_sy=self.df_sy
        filter_dfs=self.cycle_filter(*args)
        pf_all=[df_sy]+filter_dfs
        
        #周期滤波后，画图
        df_all = pd.concat(pf_all, axis=1, sort=False)
        ax_1 = df_all.plot(use_index=True
                           , figsize=(12, 6), title=u"%s高斯滤波后"%df_sy.name)
        ax_1.set_xlabel(u"日期")
        #ax_1.set_ylabel(u"单位（%）")
        ax_1.grid(True)
        # xticks = range(0, len(df_all), 6)
        # xticklabels = [df_all.index[i] for i in xticks]
        # ax_1.set_xticks(xticks)
        # ax_1.set_xticklabels(xticklabels, rotation=70)  # roration 旋转
        
        #plt.show()

        
    """
    按照指定周期高斯滤波后，进行回归拟合
    参数：周期（月）
    返回：拟合图形
    """
    def cycle_fit_by_gau(self,*args):
    
        #周期滤波
        fiter_dfs=self.cycle_filter(*args)
        if not len(fiter_dfs):
            print("周期滤波 为空")
            return
        #待拟合数据
        X =[]
        for i in range(len(fiter_dfs[0])):
            df_list=[]
            for item in fiter_dfs:
                df_list.append(item[i])
            X.append(df_list)
        
        #调用模型
        clf = linear_model.LinearRegression()
        Y = self.df_sy
        #训练模型
        clf.fit(X,Y)

        # 拟合值
        y_hat = clf.predict(X)
        
        #画图
        three_fit_df=pd.Series(y_hat,Y.index)
        three_fit_df.name="%s_fit"%Y.name

        df_all_fit = pd.concat([Y,three_fit_df ], axis=1, sort=False)
        ax_2 = df_all_fit.plot(use_index=True
                           , figsize=(12,6), title=u"%s高斯滤波后，三周期线性回归合成"%Y.name)
        ax_2.set_xlabel(u"日期")
        #ax_2.set_ylabel(u"单位（%）")
        ax_2.grid(True)
        # xticks_2 = range(0, len(df_all_fit), 6)
        # xticklabels_2 = [df_all_fit.index[i] for i in xticks_2]
        # ax_2.set_xticks(xticks_2)
        # ax_2.set_xticklabels(xticklabels_2, rotation=70)  # roration 旋转
        
        
        return three_fit_df

    """
    简单拟合
    按照2个余弦进行拟合
    """
    def cycle_simple_fit(self,t1,t2):
        df_sy=self.df_sy
        df_old=df_sy.copy()
        
        df_sy.index=range(len(df_sy))
        
        # 创建函数模型用来生成数据
        def func(x, a, b, c, d, e):
            fu = a * np.cos(np.pi * (2/ t1) * x + d) + b * np.cos(np.pi * (2 / t2) * x + e) - c
            return fu

        # 待拟合数据
        x = df_sy.index
        y = df_sy

        popt, pcov = curve_fit(func, x, y)
        df_func = []
        for i in df_sy.index:
            df_func.append(func(i, popt[0], popt[1], popt[2], popt[3], popt[4]))

        simple_fit_df = pd.Series(df_func, df_old.index)
        simple_fit_df.name = "simple_fit  %s and %s" %(t1,t2)

        df_all_fit = pd.concat([df_old, simple_fit_df], axis=1, sort=False)
        ax_2 = df_all_fit.plot(use_index=True
                               , figsize=(12, 6), title=u'简单周期模拟情况%s 和%s'%(t1,t2))
        ax_2.set_xlabel(u"日期")
        ax_2.grid(True)
        # xticks_2 = range(0, len(df_all_fit), 6)
        # xticklabels_2 = [df_all_fit.index[i] for i in xticks_2]
        # ax_2.set_xticks(xticks_2)
        # ax_2.set_xticklabels(xticklabels_2, rotation=70)  # roration 旋转

        # plt.figure(figsize=(12,5))
        # plt.title('简单周期模拟情况%s 和%s'%(t1,t2))
        # plt.xlabel('iteration times')
        # plt.ylabel('sy_rate')
        # plt.plot(df_old.index, df_func, color='green', label='new')
        # plt.plot(df_old.index, df_sy, color='red', label='original')
        #
        # plt.legend()  # 显示图例
        plt.show()

    
    


