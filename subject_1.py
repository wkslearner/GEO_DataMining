#!/usr/bin/python
# encoding=utf-8

import pandas as pd

'''数据分割'''
def split_data(dataframe,gid,raw):
    '''
    :param dataframe:
    :param gid:
    :param raw:
    :return:
    '''
    dataframe_lenght=dataframe.shape[0]

    data_list = []
    for i in range(dataframe_lenght):
        gid_data=dataframe[i:i+1][gid].values[0]
        raw_data=dataframe[i:i+1][raw].values[0]
        batch_data=raw_data.split(';')
        data_lenght=len(batch_data)
        for j in range(data_lenght):
            first_layer = batch_data[j].split(',')
            second_layer = [item for item in first_layer]
            third_layer = [item.split(':')[1] for item in second_layer]
            data_list.append([gid_data] + third_layer)

    result_df=pd.DataFrame(data_list,columns=['gid','平台类型','平台代码','注册时间'])

    return result_df


'''合并数据'''
def data_merge(first_df,merge_list,merge_key):
    '''
    :param first_df:
    :param merge_list:
    :param merge_key:
    :return:
    '''
    merge_df=first_df
    for i in range(len(merge_list)):
        merge_df=pd.merge(merge_df,merge_list[i],on=merge_key,how='left')

    return merge_df


def main():
    bank_df = pd.read_excel('题目一.xlsx')
    split_df=split_data(bank_df,'gid','raw1')
    register_num=split_df['平台代码'].groupby(split_df['gid']).count().reset_index()
    register_nobank=split_df['平台代码'].groupby([split_df['gid'],split_df['平台类型']]).count().reset_index()
    register_nobank=register_nobank[register_nobank['平台类型']=='非银行']
    register_nobank=register_nobank.drop(['平台类型'],axis=1)
    min_register=split_df['注册时间'].groupby(split_df['gid']).min().reset_index()

    result_bank_df=data_merge(bank_df[['gid','raw1']],[register_num,register_nobank,min_register],'gid')
    result_bank_df.columns=['gid','raw1','注册总次数','注册次数_非银行','最早注册时间']

    return result_bank_df

if __name__ == "__main__":
    result_bank_df=main()
    excel_writer=pd.ExcelWriter('题目一结果.xlsx',engine='xlsxwriter')
    result_bank_df.to_excel(excel_writer,'result',index=False)
    excel_writer.save()