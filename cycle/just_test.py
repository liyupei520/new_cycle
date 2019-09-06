#-*- coding:utf-8 -*-

from numpy import *

li = [[1, 1], [1, 3], [2, 3], [4, 4], [2, 4]]
matrix = mat(li).T #创建矩阵
# 求均值
mean_matrix = mean(matrix, axis=1)
# print(mean_matrix.shape)
# 减去平均值
Dataadjust = matrix - mean_matrix
print("0000",matrix,Dataadjust)
# print(Dataadjust.shape)
# 计算特征值和特征向量
covMatrix = cov(Dataadjust, rowvar=1) #协方差矩阵
print("0101",covMatrix)
eigValues, eigVectors = linalg.eig(covMatrix) #特征值，特征向量
# print(eigValues.shape)
# print(eigVectors.shape)
print("1111",eigValues)
print("222",eigVectors)
# 对特征值进行排序
eigValuesIndex = argsort(eigValues)
print(eigValuesIndex)
# 保留前K个最大的特征值
eigValuesIndex = eigValuesIndex[:-(1000000):-1]
print("eigValuesIndex",eigValuesIndex)
# 计算出对应的特征向量
trueEigVectors = eigVectors[:, eigValuesIndex]
print("trueEigVectors",trueEigVectors)
# 选择较大特征值对应的特征向量
maxvector_eigval = trueEigVectors[:,0]
print("maxvector_eigval",maxvector_eigval)
# 执行PCA变换：Y=PX 得到的Y就是PCA降维后的值 数据集矩阵
pca_result = eigVectors.T * Dataadjust
print("333333333333")
print(pca_result)
print(mean_matrix)
print(eigVectors*pca_result+mean_matrix)

index=[1,0]
print(index[:1])
a=[[2,3],
 [3,4],
 [5,4],
 [6,4]]
a=mat(a)
print(a)
print(a[index[:1]])
print (a[[1,3]])
