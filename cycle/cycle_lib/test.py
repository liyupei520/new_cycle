#-*- coding:utf-8 -*-
import  data_clean_lib as  clean_tool
import cycle_lib as cycle_tool
import pandas as pd
import matplotlib.pyplot as plt


#画图_双轴
def show_2_from_df(df_all, name_0,name_1):
    title= "%s 和 %s"%(name_0,name_1)
    df_all.columns=[name_0,name_1]
    cols=df_all.columns.tolist()
    #print(title,"相关系数：",df_all.corr().iat[0, 1])
    ax = df_all.plot(use_index=True, y=cols, figsize=(10, 6),secondary_y=cols[1:], title=title)
    ax.grid(True)


#平移获取相关系数
def get_best_corr(df,name_0,name_1):
    #print(name_0,name_1,df)
    df_corr=df.corr().iat[0, 1]
    df_corr_list=[]
    for i in range(-12,13,1):
        df_1=df.copy()
        cols = df_1.columns.tolist()[1]
        df_1[cols]=df_1[cols].shift(i)
        #print("aaaaa",i,df_1)
        df_corr_shift=df_1.corr().iat[0, 1]
        #print("bbb",i,df_corr_shift)
        if abs(df_corr_shift)>abs(df_corr)  and abs(df_corr_shift)>0.3:
            df_corr_list.append([i,df_corr_shift,df_1])
            
    title = "%s 和 %s" % (name_0, name_1)
    big_corr=[]
    for item in df_corr_list:
        if not big_corr:
            big_corr=item
        if abs(item[1])>abs(big_corr[1]):
            big_corr=item
            
    if not big_corr:
        print(title,"相关系数：",df_corr)
        return
    name_0_0=""
    name_0_0=name_1+str(big_corr[0])
    show_2_from_df(big_corr[-1], name_0, name_0_0)
    plt.savefig('C:\\Users\\li\\Desktop\\work\\cycle\\cycle_lib\\hp滤波后 周期拟合后 %s %s 相关系数原始%s   位移%s后%s.png' % (
    name_0.replace("：","").replace(":",""), name_1.replace("：","").replace(":",""),df_corr,big_corr[0],big_corr[1]), dpi=300)        
    print(title,"相关系数：",df_corr,"     存在位移更好的相关系数",big_corr[:-1])

    

if __name__ == '__main__':
    clean_tool= clean_tool.Pd_Info()
    # pf=clean_tool.data_day2month("CPI_PPI.xls","last")
    #
    # pf=pf["CPI"]
    # #pf=clean_tool.get_avg_std(pf)
    # print(pf)
    # pf=clean_tool.hp_lb(pf,14400)[1]
    # pf_cycle=cycle_tool.Cycle_PF(pf)
    #
    # pf_cycle.show_cycle_filter(42,100)
    # pf_cycle.cycle_fit_by_gau(42,100)
    # pf_cycle.cycle_simple_fit(42,20)


    pf_all=clean_tool.get_data_file("汽车行业数据汇总（年度数据）.xls")
    pf_all=clean_tool.get_data_file("汽车行业数据汇总（月度数据）填充数据.xls")
    
    #hp滤波的设定参数 年100 季度1600 月14400
    step=14400
    #pf_1=pf_all["蒜"]
    #pf = clean_tool.hp_lb(pf_1, 14400)[1]
    
    
    pf_all=pf_all.set_index(["date"])
    pf_all_new=pf_all
    
    
    pf_sz= pf_all_new.columns.tolist()[0] #上证指数
    sz_lb_pf=clean_tool.hp_lb(pf_all_new[pf_sz].dropna(),step)[1]
    
    sz_cycle=cycle_tool.Cycle_PF(sz_lb_pf)
    sz_fit_42_gau=sz_cycle.cycle_fit_by_gau(42)
    # sz_cycle=cycle_tool.Cycle_PF(sz_lb_pf)
    # sz_cycle.show_cycle_filter(42)
    
    
    for item in pf_all_new.columns.tolist()[1:]:
        #print("000000000000000000000",item)
        #原始图像相比
        clean_tool.showe_2_from_df(pf_all_new,pf_sz,item)
        plt.savefig('C:\\Users\\li\\Desktop\\work\\cycle\\cycle_lib\\%s %s.png' % (pf_sz, item.replace("：","").replace(":","")), dpi=300)
        #原始图像hp滤波
        item_lb_pf=clean_tool.hp_lb(pf_all_new[item].dropna(),step)[1]
        plt.savefig('C:\\Users\\li\\Desktop\\work\\cycle\\cycle_lib\\%s hp后.png' % (item.replace("：","").replace(":","")), dpi=300)
 
        
        #原始图像hp滤波后，周期趋势相比
        clean_tool.showe_2_from_df(pd.concat([sz_lb_pf,item_lb_pf],axis=1))   #hp滤波后
        plt.savefig('C:\\Users\\li\\Desktop\\work\\cycle\\cycle_lib\\%s %s hp后.png' % (pf_sz, item.replace("：","").replace(":","")), dpi=300)
        
       #原始图像hp滤波后,周期趋势位移
        get_best_corr(pd.concat([sz_lb_pf,item_lb_pf],axis=1),pf_sz,item)
        
        #原始图像hp滤波后，周期趋势42周期滤波，位移
        item_cycle=cycle_tool.Cycle_PF(item_lb_pf)
        item_fit_42_gau=item_cycle.cycle_fit_by_gau(42)
        clean_tool.showe_2_from_df(pd.concat([sz_fit_42_gau,item_fit_42_gau],axis=1))   #hp滤波后,42周期滤波后
        plt.savefig('C:\\Users\\li\\Desktop\\work\\cycle\\cycle_lib\\%s 42周期滤波.png' % ( item.replace("：","").replace(":","")), dpi=300)
        get_best_corr(pd.concat([sz_fit_42_gau,item_fit_42_gau],axis=1),pf_sz,item)
        
        
        
    # for item in pf_all_new.columns.tolist()[1:]:
        
        # item_lb_pf=clean_tool.hp_lb(pf_all_new[item].dropna(),100)[1]
        # item_cycle=cycle_tool.Cycle_PF(item_lb_pf)
        # item_fit_42_gau=item_cycle.cycle_fit_by_gau(42)
        # get_best_corr(pd.concat([sz_fit_42_gau,item_fit_42_gau],axis=1),pf_sz,item)
    
    
    
    #plt.show()    
        # plt.cla()
        # plt.close("all")
    
    print(pf_all_new.columns.tolist())
    
    # i=1
    # pf=pf_all_new[pf_all_new.columns.tolist()[i]]
    
    # pf_c=clean_tool.hp_lb(pf_all_new[pf_all_new.columns.tolist()[i]].dropna(),14400)[1]
    
    


