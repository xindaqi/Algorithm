import matplotlib.pyplot as plt
import numpy as np
import math

x = [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243,
     0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719, 0.359, 0.339, 0.282,
     0.748, 0.714, 0.483, 0.478, 0.525, 0.751, 0.532, 0.473, 0.725, 0.446]
# print("length of x: {}".format(len(x)))
y = [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267,
     0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103, 0.188, 0.241, 0.257,
     0.232, 0.346, 0.312, 0.437, 0.369, 0.489, 0.472, 0.376, 0.445, 0.459]
def test_matrix():
    sigma = np.mat([[0.2, 0.1], [0.0, 0.1]])
    sigma_Trans = sigma.T
    sigma_inverse = sigma.I
    print("sigma: {}".format(sigma))
    print("sigma Inverse: {}".format(sigma_inverse))
    print("sigma Transform: {}".format(sigma_Trans))


def gauss_density_probability(n, x, mu, sigma):
    sigma_det = np.linalg.det(sigma)
    divisor = pow(2*np.pi, n/2)*np.sqrt(sigma_det)
    exp = np.exp(-0.5*(x-mu)*sigma.I*(x-mu).T)
    p = exp/divisor
    return p

def test_posterior_probability():
    xx = np.mat([[x[0], y[0]]])
    mu_datasets = [np.mat([[x[5], y[5]]]), np.mat([[x[21], y[21]]]), np.mat([[x[26], y[26]]])]
    sigma = np.mat([[0.1, 0.0], [0.0, 0.1]])
    det = np.linalg.det(sigma)
    print("det: {}".format(det))
    p_all = []
    for mu in mu_datasets:
        p = gauss_density_probability(2, xx, mu, sigma)
        p = p/3
        p_all.append(p)
    p_mean = []
    for p in p_all:
        p_sum = np.sum(p_all)
        p = p/p_sum
        p_mean.append(p)
    print("probability: {}".format(p_mean[0]))

def posterior_probability(k, steps):
    alpha_datasets = [np.mat([1/k]) for _ in range(k)]
    xx = [np.mat([[x[i], y[i]]]) for i in range(len(x))]
    mu_rand = np.random.randint(0, 30, (1, k))
    print("random: {}".format(mu_rand[0]))
#     mu_datasets = [np.mat([[x[i], y[i]]]) for i in mu_rand[0]]
    mu_datasets = [np.mat([[x[5], y[5]]]), np.mat([[x[21], y[21]]]), np.mat([[x[26], y[26]]])]
    sigma_datasets = [np.mat([[0.1, 0.0], [0.0, 0.1]]) for _ in range(k)]
#     det = np.linalg.det(sigma_datasets[0])
    for step in range(steps):
        p_all = []
        # create cluster
        classification_temp = locals()
        for i in range(k):
            classification_temp['cluster_'+str(i)] = []
        # post probability  
        for j in range(len(x)):
            p_group = []
            for i in range(k):
                mu = mu_datasets[i]
                
                p = gauss_density_probability(2, xx[j], mu, sigma_datasets[i])

                p = p*alpha_datasets[i].getA()[0][0]
                p_group.append(p)
            p_sum = np.sum(p_group)
            max_p = max(p_group)
            max_index = p_group.index(max_p)
            for i in range(k):
                if i == max_index:
                    classification_temp['cluster_'+str(max_index)].append(j)
            
            p_group = [p_group[i]/p_sum for i in range(len(p_group))]
            p_all.append(p_group)
#         for i in range(k):
#             print("cluster {}:{}".format(i, classification_temp['cluster_'+str(i)]))
            

        # update mu, sigma, alpha
        mu_datasets = []
        sigma_datasets = []
        alpha_datasets = []

        for i in range(k):
            mu_temp_numerator = 0
            mu_temp_denominator = 0
            sigma_temp = 0
            alpha_temp = 0
            mu_numerator = [p_all[j][i]*xx[j] for j in range(len(x))]
            for mm in mu_numerator:
                mu_temp_numerator += mm

            mu_denominator = [p_all[j][i] for j in range(len(x))]
            for nn in mu_denominator:
                mu_temp_denominator += nn

            mu_dataset = mu_temp_numerator/mu_temp_denominator
            mu_datasets.append(mu_dataset)

            sigma = [p_all[j][i].getA()[0][0]*(xx[j]-mu_dataset).T*(xx[j]-mu_dataset) for j in range(len(x))]
            for ss in sigma:
                sigma_temp += ss
            sigma_dataset = sigma_temp/mu_temp_denominator
            sigma_datasets.append(sigma_dataset)

            alpha_new = [p_all[j][i] for j in range(len(x))]
            for alpha_nn in alpha_new:
                alpha_temp += alpha_nn
            alpha_dataset = alpha_temp/len(x)
            alpha_datasets.append(alpha_dataset)
#         print("posterior probability: {}".format(p_all[0][0].getA()))
#         print("posterior probability: {}".format(post_probability))
#         print("mu datasets: {}".format(mu_datasets))
#         print("sigma datasets: {}".format(sigma_datasets))
#         print("alpha datasets: {}".format(alpha_datasets))
#         print("-------------------")
    
    return p_all, mu_datasets, sigma_datasets, alpha_datasets, classification_temp
def cluster_visiualization(k, steps):
    post_probability, mu_datasets, sigma_datasets, alpha_datasets, classification_temp = posterior_probability(k, steps)
    plt.figure(figsize=(8, 8))
    markers = ['.', 's', '^', '<', '>', 'P']
    plt.xlim(0.1, 0.9)
    plt.ylim(0, 0.9)
    plt.grid()
    plt.scatter(x, y, color='r')
    
    plt.figure(figsize=(8, 8))
    for i in range(k):
        xx = [x[num] for num in classification_temp['cluster_'+str(i)]]
        yy = [y[num] for num in classification_temp['cluster_'+str(i)]]
        
        plt.xlim(0.1, 0.9)
        plt.ylim(0, 0.9)
        plt.grid()
        plt.scatter(xx, yy, marker=markers[i])
    plt.savefig("./images/gauss_cluster.png", format="png")
#         print("classification: {}".format(classification_temp['cluster_'+str(i)]))
#         print("cluster {} xx: {}".format(i, xx))
    
    
    
if __name__ == "__main__":
#     posterior_probability(3, 5)
#     post_probability, mu_datasets, sigma_datasets, alpha_datasets, classification_temp = posterior_probability(3, 5)
#     print("posterior probability: {}".format(post_probability[0][0].getA()[0][0]))
#     print("mu datasets: {}".format(mu_datasets))
#     print("sigma datasets: {}".format(sigma_datasets))
#     print("alpha datasets: {}".format(alpha_datasets))
#     print("classifiction: {}".format())
    cluster_visiualization(3, 100)
    
    