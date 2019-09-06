#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
import statsmodels.tsa.stattools as st
from dateutil.parser import parse
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
import tushare  as ts
import os

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号



def get_data_by_tushare(code="",flag=""):
    '''
    :param code: 股票代码
    :param flag: 数据类别 open	high	close	low	volume	price_change	p_change	ma5	ma10	ma20	v_ma5	v_ma10	v_ma20
    :return: 获取原始数据
    '''
    #ts.set_token('185214390e9d12612b4ab3499558a7e5511b8290430b6ae34cae10ad')
    if code=="":
        code="sh"
    if flag=="":
        flag="close"
    df = ts.get_hist_data(code)
    df.head(10)
    #print(df)
    #df.to_csv('%s/k/sh.csv'%os.getcwd())

    sz=df.sort_index(axis=0, ascending=True) #对index进行升序排列
    sz_return=sz[[flag]] #选取数据类型
    return sz_return

def test_stationarity(timeseries):
    # # 滑动均值和方差
    # rolmean = timeseries.rolling(4).mean()
    # rolstd = timeseries.rolling(4).std()
    #
    # # 绘制滑动统计量
    # plt.figure(figsize=(24, 8))
    # orig = plt.plot(timeseries, color='blue', label='Original')
    # mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    # std = plt.plot(rolstd, color='black', label='Rolling Std')
    #
    # plt.legend(loc='best')
    # plt.title('Rolling Mean & Standard Deviation')
    # plt.show(block=False)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111,title=u"1阶差分")
    diff1 = timeseries
    diff1.plot(ax=ax1)

    plot_acf(timeseries,lags=40,title="ACF")
    plot_pacf(timeseries,lags=40,title="PACF")
    plt.show()
    # adf检验
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used',
                                             'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def run_main():
    #原始数据
    time_data=get_data_by_tushare(flag="close")
    data=time_data["close"]
    #print("1111",data,time_data.index)
    index = [parse(i) for i in time_data.index]
    print ("2222",index)
    SZ_SERIES = pd.Series(data,index)
    SZ_SERIES.plot(label="orign",figsize=(12,6),title=u"上证指数每日收盘价")
    plt.legend(loc='best')
    plt.show()


    # 原始数据的
    #test_stationarity(SZ_SERIES)

    # 指数
    GDP_LOG = np.log(SZ_SERIES)
    test_stationarity(GDP_LOG)
    
    # 1阶差分
    SZ_DIFF1 = SZ_SERIES.diff(1)
    print("SZ_DIFF1",SZ_DIFF1)
    SZ_DIFF1.dropna(inplace=True)
    test_stationarity(SZ_DIFF1)

    # diff=""
    # for i in range(4):  # 五阶差分，一般一到二阶就行了
    #     diff = SZ_DIFF1.diff(1)
    #     diff = diff.dropna()
    # plt.figure()
    # plt.plot(diff)
    # plt.title(u'五阶差分')
    # plt.show()
    #
    # # 5阶差分的ACF
    # acf_diff = plot_acf(diff, lags=20)
    # plt.title(u"五阶差分的ACF")  # 根据ACF图，观察来判断q
    # acf_diff.show()
    #
    # # 5阶差分的PACF
    # pacf_diff = plot_pacf(diff, lags=20)  # 根据PACF图，观察来判断p
    # plt.title(u"五阶差分的PACF")
    # pacf_diff.show()

    #plot_acf(time_series_new)
    #plot_pacf(time_series_new)


    #暴力定阶
    #这里需要设定自动取阶的 p和q 的最大值，即函数里面的max_ar,和max_ma。ic 参数表示选用的选取标准，这里设置的为aic,当然也可以用bic。然后函数会算出每个 p和q 组合(这里是(0,0)~(3,3)的AIC的值，取其中最小的,这里的结果是(p=0,q=1)。

    order=st.arma_order_select_ic(SZ_DIFF1,max_ma=3,max_ar=3,ic=["aic","bic","hqic"])
    #(p, q) =(sm.tsa.arma_order_select_ic(dta,max_ar=3,max_ma=3,ic='aic')['aic_min_order'])
    (p,q) =order.aic_min_order
    print(p,q)


    #.拟合（训练）是计算各自回归项和滑动平均项的系数。
    arma_model = ARMA(SZ_DIFF1,(6,3)).fit()
    #arma_model = ARMA(SZ_DIFF1, (p, q)).fit()
    #arma_model.summary2()        #生成一份模型报告
    #arma_model.forecast(5)
    #PREDICT_ARMA=arma_model.predict('2019-04-01','2019-04-20',dynamic=True)
    PREDICT_ARMA = arma_model.predict(start=len(index)-200,end=len(index)+5,dynamic=True)

    #获取股票开盘日期
    OpenList = ts.trade_cal()
    date=index[-200]
    print (date)
    date_format = date.strftime('%Y-%m-%d')
    new_data_list = OpenList[OpenList.calendarDate > date_format]
    PREDICT_ARMA.index=pd.to_datetime(new_data_list[new_data_list['isOpen'].isin(['1'])][:206]['calendarDate'])

    #PREDICT_ARMA.index=[i+datetime.timedelta(days=1) for i in index[-7:]]
    plt.figure(figsize=(24, 8))
    orig = plt.plot(SZ_DIFF1, color='blue', label='Original')
    plt.figure(figsize=(24, 8))
    predict = plt.plot(PREDICT_ARMA, color='red', label='Predict')
    plt.legend(loc='best')
    #plt.title('Original&Predict')

    plt.show()
    print(PREDICT_ARMA)
    pre_num=5
    # 一阶差分还原
    print("111111111111",SZ_SERIES.shift(1))
    PREDICT_SZ = PREDICT_ARMA.add(SZ_SERIES.shift(1))
    PREDICT_SZ["2019-04-17"] = PREDICT_ARMA["2019-04-17"] + SZ_SERIES["2019-04-16"]
#    for i in range(1,pre_num+1):#i 为预测数量
#        PREDICT_SZ[len(SZ_DIFF1)+i] = PREDICT_ARMA[len(SZ_DIFF1)+i] + PREDICT_SZ[len(SZ_DIFF1)+i-1]

    print("end",PREDICT_ARMA)
    print("pre  PREDICT_SZ",PREDICT_SZ)
    #PREDICT_ARMA.index=pd.date_range(start='20190416',end='20190420')
    plt.figure(figsize=(24, 8))
    orig = plt.plot(SZ_SERIES, color='blue',label='Original')
    predict = plt.plot(PREDICT_SZ, color='red',label='Predict')
    plt.legend(loc='best')
    plt.title('Original&Predict')
    plt.show()



#所以可以建立ARIMA 模型，ARIMA(0,1,1)
# model = ARIMA(time_series, (p,1,q)).fit()
# model.summary2()        #生成一份模型报告
#    #为未来5天进行预测， 返回预测结果， 标准误差， 和置信区间

if __name__=="__main__":
    run_main()
