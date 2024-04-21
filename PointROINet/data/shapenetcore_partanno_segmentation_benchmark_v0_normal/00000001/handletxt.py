import numpy as np
f_path=r'C:\Users\Admin\Desktop\Pointnet3\data\shapenetcore_partanno_segmentation_benchmark_v0_normal\00000003\30.txt'
with open(f_path) as f:
    contents=f.read()
'''print(contents)'''
print(type(contents))
a = contents.split()
'''print(a)'''
a = np.array(a)
count=len(open(f_path,'rU').readlines())
print(count)
a = a.reshape(count,8)
a = a[:,0:7]
print(a[0,0])
print(type(a ))

with open(f_path, 'w') as f:
        '''按列生成矩阵'''
        '''np.savetxt(f, np.column_stack(a), fmt='%s')'''
        '''按行生成矩阵'''
        np.savetxt(f, np.row_stack(a), fmt='%s')


