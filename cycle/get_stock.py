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

plt.rcParams['font.sans-serif']=['SimHei'] #����������ʾ���ı�ǩ
plt.rcParams['axes.unicode_minus']=False #����������ʾ����



def get_data_by_tushare(code="",flag=""):
    '''
    :param code: ��Ʊ����
    :param flag: ������� open	high	close	low	volume	price_change	p_change	ma5	ma10	ma20	v_ma5	v_ma10	v_ma20
    :return: ��ȡԭʼ����
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

    sz=df.sort_index(axis=0, ascending=True) #��index������������
    sz_return=sz[[flag]] #ѡȡ��������
    return sz_return

def test_stationarity(timeseries):
    # # ������ֵ�ͷ���
    # rolmean = timeseries.rolling(4).mean()
    # rolstd = timeseries.rolling(4).std()
    #
    # # ���ƻ���ͳ����
    # plt.figure(figsize=(24, 8))
    # orig = plt.plot(timeseries, color='blue', label='Original')
    # mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    # std = plt.plot(rolstd, color='black', label='Rolling Std')
    #
    # plt.legend(loc='best')
    # plt.title('Rolling Mean & Standard Deviation')
    # plt.show(block=False)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111,title=u"1�ײ��")
    diff1 = timeseries
    diff1.plot(ax=ax1)

    plot_acf(timeseries,lags=40,title="ACF")
    plot_pacf(timeseries,lags=40,title="PACF")
    plt.show()
    # adf����
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used',
                                             'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def run_main():
    #ԭʼ����
    time_data=get_data_by_tushare(flag="close")
    data=time_data["close"]
    #print("1111",data,time_data.index)
    index = [parse(i) for i in time_data.index]
    print ("2222",index)
    SZ_SERIES = pd.Series(data,index)
    SZ_SERIES.plot(label="orign",figsize=(12,6),title=u"��ָ֤��ÿ�����̼�")
    plt.legend(loc='best')
    plt.show()


    # ԭʼ���ݵ�
    #test_stationarity(SZ_SERIES)

    # ָ��
    GDP_LOG = np.log(SZ_SERIES)
    test_stationarity(GDP_LOG)
    
    # 1�ײ��
    SZ_DIFF1 = SZ_SERIES.diff(1)
    print("SZ_DIFF1",SZ_DIFF1)
    SZ_DIFF1.dropna(inplace=True)
    test_stationarity(SZ_DIFF1)

    # diff=""
    # for i in range(4):  # ��ײ�֣�һ��һ�����׾�����
    #     diff = SZ_DIFF1.diff(1)
    #     diff = diff.dropna()
    # plt.figure()
    # plt.plot(diff)
    # plt.title(u'��ײ��')
    # plt.show()
    #
    # # 5�ײ�ֵ�ACF
    # acf_diff = plot_acf(diff, lags=20)
    # plt.title(u"��ײ�ֵ�ACF")  # ����ACFͼ���۲����ж�q
    # acf_diff.show()
    #
    # # 5�ײ�ֵ�PACF
    # pacf_diff = plot_pacf(diff, lags=20)  # ����PACFͼ���۲����ж�p
    # plt.title(u"��ײ�ֵ�PACF")
    # pacf_diff.show()

    #plot_acf(time_series_new)
    #plot_pacf(time_series_new)


    #��������
    #������Ҫ�趨�Զ�ȡ�׵� p��q �����ֵ�������������max_ar,��max_ma��ic ������ʾѡ�õ�ѡȡ��׼���������õ�Ϊaic,��ȻҲ������bic��Ȼ���������ÿ�� p��q ���(������(0,0)~(3,3)��AIC��ֵ��ȡ������С��,����Ľ����(p=0,q=1)��

    order=st.arma_order_select_ic(SZ_DIFF1,max_ma=3,max_ar=3,ic=["aic","bic","hqic"])
    #(p, q) =(sm.tsa.arma_order_select_ic(dta,max_ar=3,max_ma=3,ic='aic')['aic_min_order'])
    (p,q) =order.aic_min_order
    print(p,q)


    #.��ϣ�ѵ�����Ǽ�����Իع���ͻ���ƽ�����ϵ����
    arma_model = ARMA(SZ_DIFF1,(6,3)).fit()
    #arma_model = ARMA(SZ_DIFF1, (p, q)).fit()
    #arma_model.summary2()        #����һ��ģ�ͱ���
    #arma_model.forecast(5)
    #PREDICT_ARMA=arma_model.predict('2019-04-01','2019-04-20',dynamic=True)
    PREDICT_ARMA = arma_model.predict(start=len(index)-200,end=len(index)+5,dynamic=True)

    #��ȡ��Ʊ��������
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
    # һ�ײ�ֻ�ԭ
    print("111111111111",SZ_SERIES.shift(1))
    PREDICT_SZ = PREDICT_ARMA.add(SZ_SERIES.shift(1))
    PREDICT_SZ["2019-04-17"] = PREDICT_ARMA["2019-04-17"] + SZ_SERIES["2019-04-16"]
#    for i in range(1,pre_num+1):#i ΪԤ������
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



#���Կ��Խ���ARIMA ģ�ͣ�ARIMA(0,1,1)
# model = ARIMA(time_series, (p,1,q)).fit()
# model.summary2()        #����һ��ģ�ͱ���
#    #Ϊδ��5�����Ԥ�⣬ ����Ԥ������ ��׼�� ����������

if __name__=="__main__":
    run_main()
