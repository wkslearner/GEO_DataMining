#!/usr/bin/python
# encoding=utf-8

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import warnings
from sklearn.learning_curve import learning_curve
from sklearn.metrics import make_scorer, mean_absolute_error
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 解决windows环境下画图汉字乱码问题

warnings.filterwarnings("ignore")


'''移除列表中部分元素，生成新列表'''
def remove_list(all_list,remove_element=[]):
    '''
    :param all_list:
    :param remove_element:
    :return:
    '''
    end_list=[]
    for element in all_list:
        if element in remove_element:
            continue
        else:
            end_list.append(element)

    return end_list


def describe_data(dataframe,var_list):
    '''
    :param dataframe:
    :param var_list:
    :return:
    '''

    # var_dict={}
    # for var in var_list:
    #     describe=dataframe[var].describe()
    #     var_dict[var]=describe
    #     print(var_dict)
    #
    # describe_df=pd.DataFrame(var_dict).reset_index()
    describe_df=dataframe[var_list].describe().reset_index()
    describe_df = describe_df.rename(columns={'index': 'stat_describe'})

    return describe_df


'''变量空值检测'''
def check_nullvalue(dataframe,var_list):
    '''
    :param dataframe:
    :param var_list:
    :return:
    '''
    null_rate_list=[]
    all_num=dataframe.shape[0]
    for var in var_list:
        null_num=dataframe[dataframe[var].isnull()].shape[0]
        null_rate=null_num/all_num
        null_rate_list.append([var,null_rate])

    null_rate_df=pd.DataFrame(null_rate_list,columns=['variable','null_rate'])
    return null_rate_df


'''变量单一值检测'''
def single_value(dataframe,var_list):
    '''
    :param dataframe:
    :param var_list:
    :return:
    '''
    single_list=[]
    for var in var_list:
        var_series=dataframe[dataframe[var].notnull()][var]
        if len(set(var_series))<=1:
            single_list.append(var)

    return single_list


def get_r_square(x, y, degree):
    '''
    :param x:
    :param y:
    :param degree:
    :return:
    '''
    coeffs = np.polyfit(x, y, degree)

    # r-squared
    p = np.poly1d(coeffs)

    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)

    r_square=ssreg / sstot

    return r_square


'''最高相关度变量'''
def max_relationship(dataframe,variable_list,p_values=0.05,rsquare_limit=0.5):
    '''
    :param dataframe: 目标数据框
    :param variable_list: 所有求解的变量列表
    :return: 两两间存在相关性的变量字典
    '''
    col = list(variable_list)
    son = list(variable_list)

    relative_list = []
    for key in col:
        son.remove(key)
        max_rsquare=0
        ls=[]
        for son_key in son:
            mid_df = dataframe[(dataframe[key].notnull()) & (dataframe[son_key].notnull())]
            var_1=mid_df[key].astype(float)
            var_2 = mid_df[son_key].astype(float)
            slope, intercept, r_value, p_value, std_err = stats.linregress(var_1, var_2)
            r_square = get_r_square(var_1, var_2, 1)
            if p_value < p_values and r_square > rsquare_limit:
                if r_square>max_rsquare:
                    ls = [key, son_key, p_value, r_square]
                    max_rsquare=r_square
        if ls!=[]:
            relative_list.append(ls)

    return relative_list


'''缺失值最相似填充法,如果不满足最相似法条件的变量用均值填充'''
def likest_fillna(dataframe,var_list):
    '''
    :param dataframe:
    :param var_list:
    :return:
    '''
    null_rate_df = check_nullvalue(dataframe, dataframe.columns)
    nullvar_list = list(null_rate_df[null_rate_df['null_rate'] > 0]['variable'])
    relate_list = max_relationship(dataframe, var_list,p_values=0.1, rsquare_limit=0.1)
    relate_df = pd.DataFrame(relate_list, columns=['var1', 'var2', 'p_value', 'r_square'])

    for var in nullvar_list:
        temp_df = relate_df[relate_df['var1'] == var][['var1', 'var2']]
        if temp_df.shape[0]==0:
            dataframe[var]=dataframe[var].fillna(dataframe[var].mean())
        else:
            relate_var = temp_df['var2'].values[0]
            dataframe = dataframe.sort_values(by=relate_var, ascending=1)
            dataframe[var] = dataframe[var].fillna(method='ffill')
            dataframe[var] = dataframe[var].fillna(method='bfill')

    return dataframe


def outerlier_check(dataframe,var_list):
    for var in var_list:
        Percentile = np.percentile(dataframe[var],[0,25,50,75,100])
        IQR = Percentile[3] - Percentile[1]
        uplimit = Percentile[3]+IQR*1.5
        lowlimit = Percentile[1]-IQR*1.5

        mean_value=dataframe[var].mean()
        dataframe[var]=dataframe[var].apply(lambda x:outer_fun(x,uplimit,lowlimit,mean_value))

    return dataframe

def outer_fun(x,upvalue,lowvalue,mean_value):
    if x>upvalue or x<lowvalue:
        return mean_value
    else:
        return x

'''等频或等宽分箱法'''
def basic_split_bin(df,var,numOfSplit = 5, method = 'equal_freq'):
    '''
    :param df: 数据集
    :param var: 需要分箱的变量。仅限数值型。
    :param numOfSplit: 需要分箱个数，默认是5
    :param method: 分箱方法，'equal freq'：，默认是等频，否则是等距
    :return:分割点列表
    '''

    if method == 'equal_freq':
        notnull_df = df.loc[~df[var].isnull()]
        N = notnull_df.shape[0]
        n = int(N / numOfSplit)
        splitPointIndex = [i * n for i in range(1, numOfSplit)]
        rawValues = sorted(list(notnull_df[var]))
        maxvalue=max(notnull_df[var])
        minvalue=min(notnull_df[var])
        splitPoint = [rawValues[i] for i in splitPointIndex]
        splitPoint = sorted(list(set(splitPoint)))
        if splitPoint[0]>minvalue:
            splitPoint.insert(0,minvalue)
        splitPoint.append(maxvalue)
        return splitPoint

    elif method=='equal_wide':
        var_max, var_min = max(df[var]), min(df[var])
        interval_len = (var_max - var_min)*1.0/numOfSplit
        splitPoint = [var_min + i*interval_len for i in range(1,numOfSplit)]
        return splitPoint

    else:
        print('the method do not exist')


'''返回分箱后的数据框和映射字典'''
def split_bin_df(dataframe,var_list):
    freq_bin_dict={}
    invaild_var_list=[]
    for var in var_list:
        freq_bin_point=basic_split_bin(dataframe,var,5)
        num=len(freq_bin_point)

        if num>1:
            threshold_dict={}
            for i in range(num-1):
                before = freq_bin_point[i]
                after = freq_bin_point[i + 1]
                dataframe.loc[(dataframe[var] >= before) & (dataframe[var] < after), var] = i+1
                threshold_dict[i+1]=str(round(before,2))+'-'+str(round(after,2))

            dataframe.loc[dataframe[var] == freq_bin_point[num - 1], var ] =i+ 1
            # threshold_dict[i+2]=str(freq_bin_point[num-2])+'-'+str(freq_bin_point[num-1])
            freq_bin_dict[var] = {'split_point':freq_bin_point,'threshold':threshold_dict}
        else:
            invaild_var_list.append(var)
            freq_bin_dict[var] = {'split_point':freq_bin_point, 'var_type':'invaild_var'}

    return dataframe,freq_bin_dict


'''woe和iv值函数'''
def woe_informationvalue(dataframe,x_key,y_key):
    '''
    :param dataframe:
    :param x_key:
    :param y_key:
    :return:
    '''
    x_category=sorted(dataframe[x_key].unique())
    #print(x_category)

    x_count=dataframe[x_key].groupby([dataframe[x_key]]).count()
    good_sum = dataframe[dataframe[y_key] == 0][y_key].count()
    bad_sum = dataframe[dataframe[y_key] == 1][y_key].count()

    woe_list=[]
    information_value=0
    accumulative_bad =0;accumulative_good=0;accumulative_all=0
    woe_dict={}
    for var in x_category:
        total_count=dataframe[dataframe[x_key]==var][x_key].count()
        bad_count=dataframe[(dataframe[x_key]==var)&(dataframe[y_key]==1)][y_key].count()
        good_count=dataframe[(dataframe[x_key]==var)&(dataframe[y_key]==0)][y_key].count()

        if bad_sum==0:
            bad_distibution=0
        else:
            bad_distibution=round(bad_count/bad_sum,3)

        if good_sum==0:
            good_distibution=0
        else:
            good_distibution=round(good_count/good_sum,3)

        if bad_distibution==0:
            woe=0
        else:
            woe=np.log10(good_distibution/bad_distibution)

        dg_db=good_distibution-bad_distibution
        dg_db_woe=dg_db*woe

        information_value=information_value+dg_db_woe
        last_dict={}
        last_dict["woe"]=woe=round(woe,3)
        last_dict["iv"]=iv=round(information_value,3)
        last_dict["bad_rate"]=bad_rate=round(bad_count/(bad_count+good_count),3)

        last_dict["bad_count"]=bad_count
        last_dict["good_count"]=good_count
        last_dict["all_count"]=all_count=bad_count+good_count

        last_dict["bad_proportion"]=bad_pro=bad_distibution
        last_dict["good_proportion"] =good_pro=good_distibution

        accumulative_bad = accumulative_bad + bad_count
        accumulative_good = accumulative_good + good_count
        accumulative_all =accumulative_all + bad_count +good_count

        last_dict["acc_bad_pro"]=acc_bad_rate=round(accumulative_bad/bad_sum,3)
        last_dict["acc_good_pro"]=acc_good_rate=round(accumulative_good/good_sum,3)
        last_dict["acc_all_pro"]=acc_all_rate=round(accumulative_all/(good_sum+bad_sum),3)

        ks=round(abs(acc_bad_rate-acc_good_rate),3)
        last_dict["ks"]=ks

        woe_dict[str(var)]=last_dict
        ls=[x_key,var,bad_count,good_count,all_count,bad_pro,good_pro,
            acc_bad_rate,acc_good_rate,acc_all_rate,
            woe,iv,ks,bad_rate]

        woe_list.append(ls)

    name_list=['var','cate','bad_count','good_count','all_count','bad_pro','good_pro',
               'accumulative_bad','accumulative_good','accumulative_all',
               'woe','iv','ks','bad_rate']
    woe_iv_df=pd.DataFrame(woe_list,columns=name_list)

    return woe_iv_df


def regression_analysis(dataframe,variable_list,pvalue=0.05,rsquare_limit=0.5):
    '''
    :param dataframe:
    :param variable_list:
    :return:
    '''
    col = list(variable_list)
    son = list(variable_list)
    relative_list = []
    for key in col:
        son.remove(key)
        for son_key in son:
            mid_df = dataframe[(dataframe[key].notnull()) & (dataframe[son_key].notnull())]
            var_1=mid_df[key].astype(float);var_2 = mid_df[son_key].astype(float)
            slope, intercept, r_value, p_value, std_err = stats.linregress(var_1, var_2)
            r_square = get_r_square(var_1, var_2, 1)
            if p_value < pvalue and r_square > rsquare_limit:
                ls = [key, son_key, p_value, r_square]
                relative_list.append(ls)

    relate_df=pd.DataFrame(relative_list,columns=['var1','var2','p_value','r_square'])
    return relate_df


'''自动剔除vif大于某个阈值的变量'''
def vif_check(dataframe,var_list,limit_value=10):
    matx = np.matrix(dataframe[var_list])
    max_vif,del_var=calculate_vif(matx,var_list)
    while max_vif>=limit_value:
        var_list.remove(del_var)
        matx = np.matrix(dataframe[var_list])
        max_vif, del_var = calculate_vif(matx, var_list)

    return var_list

def calculate_vif(matx,var_list):
    num = len(var_list)
    max_vif=0
    for i in range(num):
        vif=variance_inflation_factor(matx,i)
        if vif>max_vif:
            max_vif=vif
            del_var=var_list[i]

    return max_vif,del_var




def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, scoring=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    '''
    返回值解释
    ------------
    train_sizes：指定的train_sizes大小的数组，shape为(n_ticks, ).
    train_scores：训练集上的得分，shape为(n_ticks, n_cv_folds)，n_cv_folds为cv的份数，默认为3
    test_scores： 测试集上的得分
    '''
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        # plt.figure()
        plt.title(title, fontproperties=font)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数", fontproperties=font)
        plt.ylabel(u"得分", fontproperties=font)
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=r"score of training set")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=r'score of cv set')

        plt.legend(loc="best")
        # plt.ion()
        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()
        # time.sleep(1)

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff




'''-------------------------------这里开始数据处理及建模-------------------------------------'''

model_data = pd.read_excel('题目三.xlsx')
data_columns=list(model_data.columns)
var_list=remove_list(data_columns,['gid','apply_time','y'])


'''小问题1——数据的描述统计'''
md_describe=describe_data(model_data,var_list)
print('小问题1结果如下：')
print(md_describe.head())
print('\n')


'''小问题2——按月统计相关数据'''
data_2=model_data.copy()
data_2['apply_mon']=data_2['apply_time'].map(lambda x:pd.Period(x,freq='M'))
stat_data_2=data_2['gid'].groupby([data_2['apply_mon'],data_2['y']]).count().reset_index()
bad_num=stat_data_2[stat_data_2['y']==1]['gid'].groupby(stat_data_2['apply_mon']).sum()
good_num=stat_data_2[stat_data_2['y']==0]['gid'].groupby(stat_data_2['apply_mon']).sum()
all_num=stat_data_2['gid'].groupby(stat_data_2['apply_mon']).sum()
bad_rate=bad_num/all_num
print('小问题2结果如下：')
print( 'bad人数:','\n',bad_num.reset_index(),'\n',
       'good人数:','\n',good_num.reset_index(),'\n',
       '总人数:','\n',all_num.reset_index(),'\n',
       'bad_rate:','\n',bad_rate.reset_index())
print('\n')


'''小问题3——建模前的数据处理'''
#空值检测,并把空值率大于30%的变量剔除,剩余64个变量
null_rate_df=check_nullvalue(model_data,var_list)
valid_var=null_rate_df[null_rate_df['null_rate']<0.3]['variable']

#单一值检测，只有一个值的变量剔除4个
single_value_var=single_value(model_data[valid_var],valid_var)
valid_var=remove_list(valid_var,single_value_var)



#日期变量全部与申请时间做差值计算
time_var=['x213','x214','x221','x222','x433','x434','x441','x442']
data3=model_data.copy()
data3=data3[['gid','apply_time','y']+valid_var]

def date_sub_fun(x,y):
    if x is np.nan or y is np.nan :
        pass
    else:
        x=pd.datetime.strptime(str(x),'%Y-%m-%d')
        y=pd.datetime.strptime(str(y),'%Y-%m-%d')
        return (x-y).days

for var in time_var:
    data3[var]=data3.apply(lambda x:date_sub_fun(x['apply_time'],x[var]),axis=1)


#缺失值填充
data3=likest_fillna(data3,valid_var)
#print(data3[data3.isnull().values==True])

# data3 = pd.read_excel('temp.xlsx')
# valid_var=remove_list(list(data3.columns),['gid','apply_time','y'])

#异常值处理
data3=outerlier_check(data3,valid_var)


'''数据分箱'''
bin_data,bin_dict=split_bin_df(data3,valid_var)


'''求解woe、iv、ks及bad_rate等值'''
woe_iv_dict={}
woe_iv_df=pd.DataFrame()
for var in valid_var:
    woe_iv=woe_informationvalue(bin_data,var,'y')
    woe_iv_df=pd.concat([woe_iv_df,woe_iv])

print('woe、iv、ks及bad_rate等值')
print(woe_iv_df.head())
print('\n')

#变量iv值排序
var_iv=woe_iv_df['iv'].groupby(woe_iv_df['var']).sum().reset_index()
var_iv=var_iv.sort_values(by='iv',ascending=0)


#变量相关性检验,排除相关性大于0.7同时iv值较小的变量
relate_df=regression_analysis(bin_data,valid_var,rsquare_limit=0.7)

del_list=[]
for i in range(relate_df.shape[0]):
    slice=relate_df[['var1','var2']][i:i+1]
    var1_iv=var_iv[var_iv['var']==slice['var1'].values[0]]['iv'].values[0]
    var2_iv = var_iv[var_iv['var'] == slice['var2'].values[0]]['iv'].values[0]

    if var1_iv>var2_iv:
        del_list.append(slice['var2'].values[0])
    else:
        del_list.append(slice['var1'].values[0])

del_list=list(set(del_list))
locgit_var_list=remove_list(valid_var,del_list)


#共线性检验，剔除方差膨胀因子大于3的变量
locgit_var_list=vif_check(bin_data,locgit_var_list,limit_value=3)



'''小问题4——使用3种方法进行模型构建'''

#划分数据集
x_train,x_test,y_train,y_test= train_test_split(bin_data[locgit_var_list],bin_data['y'],
                                                test_size=0.25,random_state=1)


'''使用逻辑回归建模'''
LR=LogisticRegression()
LR.fit(x_train,y_train)


print('逻辑回归精度信息')
y_train_pred_logit = LR.predict(x_train)
y_test_pred_logit = LR.predict(x_test)
acc_train = accuracy_score(y_train, y_train_pred_logit)
acc_test = accuracy_score(y_test, y_test_pred_logit)


print('logit regression train/test accuracies %.3f/%.3f' % (acc_train, acc_test))
print('logit regression train/test auc %.3f/%.3f' % (roc_auc_score(y_train,y_train_pred_logit),
                                                roc_auc_score(y_test, y_test_pred_logit)))
print('logit regression train/test Recall %.3f/%.3f' % (recall_score(y_train,y_train_pred_logit),
                                                   recall_score(y_test, y_test_pred_logit)))
print('logit regression train/test precision %.3f/%.3f' % (precision_score(y_train,y_train_pred_logit),
                                                      precision_score(y_test, y_test_pred_logit)))




'''使用xgboost建模'''

#超参数搜索
# parameter_tree = {'max_depth': range(5, 10),'n_estimators':range(100,200,10),
#                   'subsample':[ i*0.1 for i in range(5,10)],
#                   'learning_rate':[i*0.03 for i in range(1,5)]}
# clf=GridSearchCV(XGBClassifier(),parameter_tree,cv=5,
#                        scoring=make_scorer(recall_score))
# clf.fit(x_train,y_train)
# best_paramter=clf.best_params_
# print(best_paramter)


XGC=XGBClassifier(n_estimators=190,max_depth=9,learning_rate=0.03,subsample=0.9)
#XGC.fit(x_train,y_train)


'''画学习曲线'''
mean_absolute_error_score = make_scorer(mean_absolute_error)
plot_learning_curve(XGC, u"学习曲线", x_train, y_train,scoring=mean_absolute_error_score)

# print('XGBoost精度信息')
# y_train_pred_xgb = XGC.predict(x_train)
# y_test_pred_xgb = XGC.predict(x_test)
# tree_train = accuracy_score(y_train, y_train_pred_xgb)
# tree_test = accuracy_score(y_test, y_test_pred_xgb)
#
#
# print('XG Boosting train/test accuracies %.3f/%.3f' % (tree_train, tree_test))
# print('XG Boosting train/test auc %.3f/%.3f' % (roc_auc_score(y_train,y_train_pred_xgb),
#                                                 roc_auc_score(y_test, y_test_pred_xgb)))
# print('XG Boosting train/test Recall %.3f/%.3f' % (recall_score(y_train,y_train_pred_xgb),
#                                                    recall_score(y_test, y_test_pred_xgb)))
# print('XG Boosting train/test precision %.3f/%.3f' % (precision_score(y_train,y_train_pred_xgb),
#                                                       precision_score(y_test, y_test_pred_xgb)))
#
#
#
#
# '''使用深度神经网络建模'''
#
# feature_columns = [tf.contrib.layers.real_valued_column("", dimension=26)]
# DNN = tf.contrib.learn.DNNClassifier(
#     feature_columns=feature_columns,hidden_units=[10, 10, 10,10],n_classes=2)
#
# DNN.fit(x=x_train,y=y_train,steps=1000)
#
#
# print('神经网络精度信息')
# y_train_pred_dnn = list(DNN.predict(x_train))
# y_test_pred_dnn = list(DNN.predict(x_test))
# acc_train = accuracy_score(y_train, y_train_pred_dnn)
# acc_test = accuracy_score(y_test, y_test_pred_dnn)
#
#
# print('neural network train/test accuracies %.3f/%.3f' % (acc_train, acc_test))
# print('neural network train/test auc %.3f/%.3f' % (roc_auc_score(y_train,y_train_pred_dnn),
#                                                 roc_auc_score(y_test, y_test_pred_dnn)))
# print('neural network train/test Recall %.3f/%.3f' % (recall_score(y_train,y_train_pred_dnn),
#                                                    recall_score(y_test, y_test_pred_dnn)))
# print('neural network train/test precision %.3f/%.3f' % (precision_score(y_train,y_train_pred_dnn),
#                                                       precision_score(y_test, y_test_pred_dnn)))

