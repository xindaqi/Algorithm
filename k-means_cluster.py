import matplotlib.pyplot as plt
import numpy as np
import math

x = [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243,
     0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719, 0.359, 0.339, 0.282,
     0.748, 0.714, 0.483, 0.478, 0.525, 0.751, 0.532, 0.473, 0.725, 0.446]
print("length of x: {}".format(len(x)))
y = [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267,
     0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103, 0.188, 0.241, 0.257,
     0.232, 0.346, 0.312, 0.437, 0.369, 0.489, 0.472, 0.376, 0.445, 0.459]
print("length of x: {}".format(len(y)))

plt.figure(figsize=(8, 8))
plt.scatter(x, y, color='r')
plt.xlim(0.1, 0.9)
plt.ylim(0, 0.9)
plt.xlabel("x data")
plt.ylabel("y data")
plt.grid(True)
plt.savefig("./images/source_data.png", format='png')
coordinate = [data for data in zip(x, y)]
print("coordinate: {}".format(coordinate))
print("data 0 x: {}".format(coordinate[0][0]))
print("data 0 y: {}".format(coordinate[0][1]))
rand = np.random.randint(0, 30)
print("random number: {}".format(rand))
init = coordinate[rand]

def direct_line():
    print("---------\n")

def class_num(k, steps):
    '''Clustering data in k classes.'''
    init_num = np.random.randint(0, 30, (1, k))
    dis_all = []
    print("initial number: {}".format(init_num))
    init_data = [coordinate[i] for i in init_num[0]]
    print("initial data: {}".format(init_data))
    '''
    Dynamic create classification list which store corresponding cluster data.
    One can operate list classification_temp['cluster_0'],..., classification_temp['cluster_(k-1)']
    '''
    for step in range(steps):
        classification_temp = locals()
        for i in range(k):
            classification_temp['cluster_' + str(i)] = []    

        for j in range(len(coordinate)):
            dis_temp = []
            for i in range(len(init_data)):
                dis = math.pow(init_data[i][0]-coordinate[j][0], 2) + math.pow(init_data[i][1]-coordinate[j][1], 2)
                dis = math.sqrt(dis)
                dis_temp.append(dis)
            dis_min = min(dis_temp)
            dis_index = dis_temp.index(dis_min)
            for i in range(k):
                '''Adding data to croresponding cluster.'''
                if i == dis_index:
                    classification_temp['cluster_'+str(dis_index)].append(j)

    #         direct_line()
    #         print("k distance: {}".format(dis_temp))
    #         print("minimum distance: {}".format(dis_min))
    #         print("minimum index: {}".format(dis_index))

    #     '''plot final results.'''
    


    #     print("distance: {}".format(dis_temp))

#         dis = math.pow(init_data[0][0]-coordinate[0][0], 2) + math.pow(init_data[0][1]-coordinate[0][1], 2)
#         dis = math.sqrt(dis)
#         print("distance: {}".format(dis))

        for i in range(k):
            xx = []
            yy = []
            for index in classification_temp['cluster_'+str(i)]:
                xx.append(coordinate[index][0])
                yy.append(coordinate[index][1])   
            xx_mean = np.mean(xx)
            yy_mean = np.mean(yy)
            if xx_mean != init_data[i][0] or yy_mean != init_data[i][0]:
                init_data[i]= (xx_mean, yy_mean)
    print("cluster center: {}".format(init_data))
                
    '''plot final results.'''
    plt.figure(figsize=(8, 8))
    plt.xlim(0.1, 0.9)
    plt.ylim(0, 0.9)
    plt.xlabel("x data")
    plt.ylabel("y data")
    plt.grid(True)
    for i in range(k):
        direct_line()
        markers = ['.', 's', '^', '<', '>', 'P']
        print("cluster {}: data: {}".format(i, classification_temp['cluster_'+str(i)]))
        xx = []
        yy = []
        for index in classification_temp['cluster_'+str(i)]:
            xx.append(coordinate[index][0])
            yy.append(coordinate[index][1])
#         print("xx: {}".format(xx))
#         print("yy: {}".format(yy))

        plt.scatter(xx, yy, marker=markers[i])
        plt.scatter(init_data[i][0], init_data[i][1], marker=markers[-1], linewidths=1, color='r')
        plt.savefig("./images/k-mean_cluster.png", format="png")
    return init_data, classification_temp

def pow_test(data):
    result = math.pow(data, 2)
    return result
if __name__ == "__main__":
    k = 5
    center, cluster = class_num(k, 50)
    for i in range(k):
        print("cluster {} data: {}".format(i, cluster['cluster_'+str(i)]))
    print("center: {}".format(center))
    
# dis = 
# for data in zip(x, y):
#     print("coordinate: {}".format(data))
#     print("data_x: {}".format(data[0]))
# for i, j in zip(x, y):
#     print("({}, {})".format(i, j))
