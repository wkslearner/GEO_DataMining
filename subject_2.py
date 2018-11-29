#!/usr/bin/python
# encoding=utf-8

import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize,read_json,to_json

json_df=pd.read_excel('题目二.xlsx')


'''解析json并转换成dataframe'''
def json_to_dataframe(dataframe,primary_key='',key_list=[],json_key=''):
    '''
    :param dataframe:
    :param primary_key:
    :param key_list:
    :param json_key:
    :return:
    '''
    key_df=dataframe[key_list]
    target_data=dataframe[json_key].values[0]

    if isinstance(target_data,list):
        json_key_df=key_df[primary_key]
        for i in range(len(target_data)):
            json_data=target_data[i]
            json_df = json_normalize(json_data)
            json_key_df = pd.concat([json_key_df, json_df], axis=1)
    else:
        json_data = json.loads(target_data)
        json_df=json_normalize(json_data)
        json_key_df = pd.concat([key_df[primary_key], json_df], axis=1)

    result_df=pd.merge(key_df,json_key_df,on=primary_key,how='left')

    return result_df


'''对层级数据进行分解'''
def layer_analysis(dataframe,key):
    '''
    :param dataframe:
    :param key:
    :return:
    '''
    df_columns=list(dataframe.columns)

    if len(dataframe[key].values[0])==0:
        final_df=dataframe
    else:
        df_columns.remove(key)
        final_df=pd.DataFrame()
        for i in range(len(dataframe[key].values[0])):
            split_df = dataframe[df_columns]
            split_df['json'] = json.dumps(dataframe[key].values[0][i])
            turn_df=json_to_dataframe(split_df,'gid',df_columns,'json')
            final_df=pd.concat([final_df,turn_df])

    return final_df


def get_data1():
    print('开始解析data1')
    json_lenght=json_df.shape[0]
    final_merge_df=pd.DataFrame()

    for i in range(json_lenght):

        data1=json_df[i:i+1][['gid','data1']].reset_index()
        first_layer_df=json_to_dataframe(data1,'gid',['gid'],'data1')
        first_layer_columns=list(first_layer_df.columns)
        first_layer_columns.remove('data.RSL')

        if len(first_layer_df['data.RSL'].values[0])==0:
            final_df=first_layer_df
        else:
            second_layer_df=json_to_dataframe(first_layer_df,'gid',first_layer_columns,'data.RSL')
            second_layer_columns=list(second_layer_df.columns)
            second_layer_columns.remove('RS.desc')
            if len(second_layer_df['RS.desc'].values[0])==0:
                final_df = second_layer_df
            else:
                third_layer_df=json_to_dataframe(second_layer_df,'gid',second_layer_columns,'RS.desc')
                final_df=third_layer_df

        final_merge_df=pd.concat([final_merge_df,final_df])

    final_merge_df = pd.merge(pd.DataFrame(json_df['gid'],columns=['gid']), final_merge_df, on='gid', how='left')
    return final_merge_df


def get_data2():
    print('开始解析data2')
    json_lenght = json_df.shape[0]
    first_json=pd.DataFrame();second_json=pd.DataFrame();third_json = pd.DataFrame()
    for i in range(json_lenght):

        data2 = json_df[i:i + 1][['gid', 'data2']].reset_index()
        split_data = data2['data2'].values[0].split(';')

        for len_splitdata in range(len(split_data)):
            split_df = pd.DataFrame()
            split_df['gid'] = data2['gid']

            if split_data[len_splitdata] != '':
                split_df['json'] = split_data[len_splitdata]

                first_layer_df = json_to_dataframe(split_df, 'gid', ['gid'], 'json')
                first_layer_columns = list(first_layer_df.columns)

                if 'header.ret_code' in first_layer_columns :
                    first_json=pd.concat([first_json,first_layer_df])

                if 'RESULTS' in first_layer_columns:
                    second_layer_df = layer_analysis(first_layer_df, 'RESULTS')
                    third_concat_df = pd.DataFrame()
                    if 'DATA' in list(second_layer_df.columns):
                        for second_df_lenght in range(second_layer_df.shape[0]):
                            second_range_df = second_layer_df[second_df_lenght:second_df_lenght + 1]
                            third_layer_df = layer_analysis(second_range_df, 'DATA')
                            third_concat_df=pd.concat([third_concat_df,third_layer_df])
                    else:
                        third_concat_df=pd.concat([third_concat_df,second_layer_df])
                    second_json=pd.concat([second_json,third_concat_df])

                if 'code' in first_layer_columns:
                    third_json = pd.concat([third_json, first_layer_df])

    final_merge_df = pd.merge(pd.DataFrame(json_df['gid'], columns=['gid']), first_json, on='gid', how='left')
    final_merge_df = pd.merge(final_merge_df, second_json, on='gid', how='left')
    final_merge_df = pd.merge(final_merge_df, third_json, on='gid', how='left')
    return final_merge_df



def get_data3():
    print('开始解析data3')
    json_lenght = json_df.shape[0]
    final_merge_df = pd.DataFrame()

    for i in range(json_lenght):
        data3 = json_df[i:i + 1][['gid', 'data3']].reset_index()
        if data3['data3'].values[0] is np.nan:
            pass
        else:
            split_data = data3['data3'].values[0].split(';')

        final_df = pd.DataFrame()
        final_df['gid'] = data3['gid']
        for len_splitdata in range(len(split_data)):
            split_df = pd.DataFrame()
            split_df['gid'] = data3['gid']

            if split_data[len_splitdata] == '':
                pass
            else:
                split_df['json'] = split_data[len_splitdata]
                first_layer_df = json_to_dataframe(split_df, 'gid', ['gid'], 'json')
                final_df = pd.merge(final_df, first_layer_df, on='gid', how='left')

        final_merge_df = pd.concat([final_merge_df, final_df])

    final_merge_df = pd.merge(pd.DataFrame(json_df['gid'], columns=['gid']), final_merge_df, on='gid', how='left')
    return final_merge_df


def get_data4():
    print('开始解析data4')
    json_lenght = json_df.shape[0]
    final_merge_df = pd.DataFrame()
    for i in range(json_lenght):
        data4 = json_df[i:i + 1][['gid', 'data4']].reset_index()

        first_layer_df = json_to_dataframe(data4, 'gid', ['gid'], 'data4')
        first_layer_columns=list(first_layer_df.columns)
        if first_layer_df['INDX304000'].values[0]!='查无此记录':
            first_layer_columns.remove('INDX304000')
            second_layer_df=json_to_dataframe(first_layer_df,'gid',first_layer_columns,'INDX304000')
            final_merge_df=pd.concat([final_merge_df,second_layer_df])
        else:
            final_merge_df = pd.concat([final_merge_df,first_layer_df])

    final_merge_df = pd.merge(pd.DataFrame(json_df['gid'], columns=['gid']), final_merge_df, on='gid', how='left')
    return final_merge_df


if __name__ == "__main__":
    data1=get_data1()
    data2=get_data2()
    data3=get_data3()
    data4=get_data4()

    excel_writer=pd.ExcelWriter('题目二结果.xlsx',engine='xlsxwriter')
    data1.to_excel(excel_writer,'data1',index=False)
    data2.to_excel(excel_writer,'data2',index=False)
    data3.to_excel(excel_writer,'data3',index=False)
    data4.to_excel(excel_writer,'data4',index=False)
    excel_writer.save()