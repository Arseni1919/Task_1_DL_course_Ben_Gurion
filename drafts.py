import PACKAGES
from Main2 import *

# with open('train_data_rep', 'rb') as fileObject:
#     loaded_data = pickle.load(fileObject)

# b = [5,3,1]

plot_graph('Orientation vs S (Poly)', legend_labels='S', x_label='Orientation', kernel='Poly')
#

# with open('Gamma vs C (RBF)', 'rb') as fileObject:
#    s1 = pickle.load(fileObject)
# with open('C (Poly)', 'rb') as fileObject:
#    s2 = pickle.load(fileObject)
# with open('C (RBF)', 'rb') as fileObject:
#    s3 = pickle.load(fileObject)
#
# name = 'C'
# plt.plot(x[j], y[j], label='%s' % x_j_labels[j])

# plt.style.use('bmh')
# plt.plot(s1['x'], s1['y'], label='Linear')
# plt.plot(s2['x'], s2['y'], label='Poly')
# plt.plot(s3['x'], s3['y'], label='RBF')
# plt.xlabel('%s Value' % name)
# plt.ylabel('Validation Accuracy')
# plt.title('Validation Accuracy vs %s' % name)
# plt.legend()
#
# plt.show()


# 1 / 99  Values  50  and  9 : get accuracy of  0.64
# 2 / 99  Values  50  and  10 : get accuracy of  0.63
# 3 / 99  Values  50  and  11 : get accuracy of  0.635
# 4 / 99  Values  50  and  20 : get accuracy of  0.615
# 5 / 99  Values  50  and  30 : get accuracy of  0.635
# 6 / 99  Values  50  and  40 : get accuracy of  0.61
# 7 / 99  Values  50  and  50 : get accuracy of  0.635
# 8 / 99  Values  50  and  60 : get accuracy of  0.605
# 9 / 99  Values  50  and  70 : get accuracy of  0.61
# 10 / 99  Values  50  and  80 : get accuracy of  0.595
# 11 / 99  Values  50  and  90 : get accuracy of  0.5750000000000001
# 12 / 99  Values  75  and  9 : get accuracy of  0.665
# 13 / 99  Values  75  and  10 : get accuracy of  0.64
# 14 / 99  Values  75  and  11 : get accuracy of  0.65
# 15 / 99  Values  75  and  20 : get accuracy of  0.64
# 16 / 99  Values  75  and  30 : get accuracy of  0.625
# 17 / 99  Values  75  and  40 : get accuracy of  0.645
# 18 / 99  Values  75  and  50 : get accuracy of  0.62
# 19 / 99  Values  75  and  60 : get accuracy of  0.6
# 20 / 99  Values  75  and  70 : get accuracy of  0.6000000000000001
# 21 / 99  Values  75  and  80 : get accuracy of  0.615
# 22 / 99  Values  75  and  90 : get accuracy of  0.61
# 23 / 99  Values  100  and  9 : get accuracy of  0.6649999999999999
# 24 / 99  Values  100  and  10 : get accuracy of  0.6599999999999999
# 25 / 99  Values  100  and  11 : get accuracy of  0.6649999999999999
# 26 / 99  Values  100  and  20 : get accuracy of  0.645
# 27 / 99  Values  100  and  30 : get accuracy of  0.645
# 28 / 99  Values  100  and  40 : get accuracy of  0.625
# 29 / 99  Values  100  and  50 : get accuracy of  0.65
# 30 / 99  Values  100  and  60 : get accuracy of  0.625
# 31 / 99  Values  100  and  70 : get accuracy of  0.635
# 32 / 99  Values  100  and  80 : get accuracy of  0.64
# 33 / 99  Values  100  and  90 : get accuracy of  0.615
# 34 / 99  Values  115  and  9 : get accuracy of  0.6799999999999999
# 35 / 99  Values  115  and  10 : get accuracy of  0.68
# 36 / 99  Values  115  and  11 : get accuracy of  0.69
# 37 / 99  Values  115  and  20 : get accuracy of  0.6799999999999999
# 38 / 99  Values  115  and  30 : get accuracy of  0.6749999999999999
# 39 / 99  Values  115  and  40 : get accuracy of  0.65
# 40 / 99  Values  115  and  50 : get accuracy of  0.64
# 41 / 99  Values  115  and  60 : get accuracy of  0.645
# 42 / 99  Values  115  and  70 : get accuracy of  0.64
# 43 / 99  Values  115  and  80 : get accuracy of  0.63
# 44 / 99  Values  115  and  90 : get accuracy of  0.635
# 45 / 99  Values  125  and  9 : get accuracy of  0.7050000000000001
# 46 / 99  Values  125  and  10 : get accuracy of  0.7200000000000001 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 47 / 99  Values  125  and  11 : get accuracy of  0.72 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 48 / 99  Values  125  and  20 : get accuracy of  0.7000000000000001
# 49 / 99  Values  125  and  30 : get accuracy of  0.6849999999999999
# 50 / 99  Values  125  and  40 : get accuracy of  0.7050000000000001
# 51 / 99  Values  125  and  50 : get accuracy of  0.6950000000000001
# 52 / 99  Values  125  and  60 : get accuracy of  0.6699999999999999
# 53 / 99  Values  125  and  70 : get accuracy of  0.68
# 54 / 99  Values  125  and  80 : get accuracy of  0.6799999999999999
# 55 / 99  Values  125  and  90 : get accuracy of  0.6799999999999999
# 56 / 99  Values  135  and  9 : get accuracy of  0.7000000000000001
# 57 / 99  Values  135  and  10 : get accuracy of  0.7050000000000001
# 58 / 99  Values  135  and  11 : get accuracy of  0.7200000000000001 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 59 / 99  Values  135  and  20 : get accuracy of  0.695
# 60 / 99  Values  135  and  30 : get accuracy of  0.7
# 61 / 99  Values  135  and  40 : get accuracy of  0.68
# 62 / 99  Values  135  and  50 : get accuracy of  0.685
# 63 / 99  Values  135  and  60 : get accuracy of  0.6799999999999999


