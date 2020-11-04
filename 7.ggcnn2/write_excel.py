#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 14:37:13 2020

@author: ldh
"""

import random
import string
import xlwt
#注意这里的 excel 文件的后缀是 xls 如果是 xlsx 打开是会提示无效,新建excel表格后要选择文本格式保存
all_str = string.ascii_letters + string.digits
excelpath =('time_result.xls')  #新建excel文件
workbook = xlwt.Workbook(encoding='utf-8')  #写入excel文件
sheet = workbook.add_sheet('Sheet1',cell_overwrite_ok=True)  #新增一个sheet工作表
headlist=[u'账号',u'密码',u'邮箱']   #写入数据头
row=0
col=0
for head in headlist:
    sheet.write(row,col,head)
    row = row+1
for i in range(1,4):#写入3行数据
    workbook.save(excelpath) #保存
    print(u"生成第[%d]个账号"%(i))