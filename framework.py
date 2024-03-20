# A program for comparing different clustering methods
# Created for publishing articles

# by Xinghao Niu
# Ph.D. student
# Department of Economic information science
# Faculty of Economics
# Moscow State University
# email: niuxh1989@gmail.com

# Methods can be choosen:
# 'k-means clustering'
# 'mean shift clustering'
# 'DBSCAN clustering'
# 'agglomerative clustering'
# 'self-organizing map clustering'
# 'gaussian mixture model clustering'

import time
import numpy as np

import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn_som.som import SOM
import sklearn.mixture

from PIL import Image
import os

class settings_a: # set parameters
    def __init__(self, directory, file):
        self.directory = directory
        os.chdir(self.directory)
        self.original_image = Image.open(file)
        image_1 = np.array(self.original_image, dtype=np.float64) /255
        h, w, d = image_1.shape
  _2 
        
        # k means clustering
        self.number_of_clusters = 20
        self.random_state = 0
        self.init = 'k-means++'
        
        # mean shift clustering
        # none
        
        # DBSCAN clustering
        self.eps_1 = 0.1
        self.min_samples_1 = 5
        
        # agglomerative clustering
        self.number_of_clusters_agg = 10
        
        # self-organizing map clustering
        self.m_1 = 10
        self.n_1 = 2
        self.dim_1 = 3
        
        # gaussian mixture model clustering
        self.number_of_components = 100
        self.random_state_1 = 0
        
class compare:
    def __init__(self, methods): 
        print('specify directory')
        directory = eval(input())
        print('\n')
        print('specify file')
        file = eval(input())
        self.settings = settings_a(directory, file)
        self.t_r = total_result(directory, self.settings)
        self.image_2 =self.settings.image_2
        self.image = 0
        self.im = 0
        self.time_total = 0
        self.clustering = 0
        self.labels = 0
        self.selected_methods = methods
        self.run()
    
    def k_means(self):
        self.record_time_start()
        self.clustering = KMeans(n_clusters=self.settings.number_of_clusters, random_state=self.settings.random_state, init = self.settings.init).fit(self.image_2)
        self.plot_1()
        self.save_result()
    
    def mean_shift(self):
        self.record_time_start()
        self.clustering = MeanShift().fit(self.image_2)
        self.labels = self.clustering.labels_
        self.record_time_end()
        self.record_time_total()
        self.plot_2()
        self.save_result()
        
    def dbscan(self):
        self.record_time_start()
        self.clustering = DBSCAN(eps=self.settings.eps_1, min_samples=self.settings.min_samples_1).fit(self.image_2)
        self.labels = self.clustering.labels_
        self.record_time_end()
        self.record_time_total()
        self.plot_2()
        self.save_result()
        
    def self_organizing_map(self):
        self.record_time_start()
        self.clustering = SOM(m = self.settings.m_1, n = self.settings.n_1, dim = self.settings.dim_1)
        self.clustering.fit(self.image_2)
        self.labels = self.clustering.predict(self.image_2)
        self.record_time_end()
        self.record_time_total()
        self.plot_2()
        self.save_result()
        
    def gaussian_mixture_model(self):
        self.record_time_start()
        self.plot_2()
        self.save_result()
        
    def run(self):
        for method in self.selected_methods:
            if method == 'k-means clustering':
                self.name = 'k-means clustering'
                self.k_means()
            if method == 'mean shift clustering':
                self.name = 'mean shift clustering'
                self.mean_shift()
            if method == 'DBSCAN clustering':
                self.name = 'DBSCAN clustering'
                self.dbscan()
            if method == 'agglomerative clustering':  
                self.name = 'agglomerative clustering'
                self.agglom()
                self.name = 'gaussian mixture model clustering'
                self.gaussian_mixture_model()   
                
    def record_time_start(self):
        self.time_start = time.time()
    
    def record_time_end(self):
        self.time_end = time.time()
    
    def record_time_total(self):
        self.time_total =  self.time_end - self.time_start
        self.t_r.storage.append(self.time_total)
        self.t_r.storage_name.append(self.name)
        
    def plot_1(self): 
        self.image = np.zeros((self.settings.h, self.settings.w, 3),dtype=np.uint8)
        for m in range(0,self.settings.h):
            for n in range(0,self.settings.w):
                self.image[m,n] = list(self.clustering.cluster_centers_[self.labels[m*self.settings.w + n]]*255)
        self.im = Image.fromarray(self.image)
        
        
    def plot_2(self):
        self.image = np.zeros((self.settings.h, self.settings.w, 3),dtype=np.uint8)
        label_list = list(set(self.labels)).copy()
        for label in label_list:
            positions = []
            for i in range(0, len(self.labels)):
                if self.labels[i]== label:
                    positions.append(list(divmod(i,self.settings.w)))
    
            sum_array = np.array([0, 0, 0])
            for position in positions: 
                sum_array = self.settings.image_1[position[0],position[1]]*255 + sum_array
    
            average_array = sum_array/len(positions)
            for position in positions: 
                self.image[position[0],position[1]] = list(average_array)
        self.im = Image.fromarray(self.image)    
        
        
    def save_result(self):
        os.chdir(self.settings.directory)
        self.im.save(self.name+'.jpg')
        
class total_result:
    def __init__(self,directory,settings):
        self.storage = []
        self.storage_name = []
        self.directory = directory
        self.settings_b = settings
    def show_time(self):
        print('\n')
        print('number of pixels:'+' '+ str(self.settings_b.h*self.settings_b.w))
        for i in range(0,len(self.storage)):
            print(self.storage_name[i] + ':'+ ' '+ str(self.storage[i]) + ' seconds'+',  '+str(self.storage[i]/60)+ ' minutes')
        print('\n')
        print(f'clustering results are stored at {self.directory}')


print('\n')
print('methods chosen:')
methods = ['k-means clustering',
            'mean shift clustering',
            'DBSCAN clustering',
            'agglomerative clustering',
            'self-organizing map clustering',
            'gaussian mixture model clustering']
for method_i in methods:
    print(method_i)
print('\n')
    
start = compare(methods)
start.t_r.show_time()

