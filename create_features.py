
# coding: utf-8

# In[4]:


BAG_Panden = '/home/data/citycentre/BAG_Panden.shp'
CIR = '/home/data/processing/CIR_2015_10.tif'


# # Dependencies

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
from PIL import Image
import os, glob
import pandas as pd
from shutil import copyfile
import matplotlib.image as mpimg
import numpy
import geopandas as gpd
import fiona
import rasterio
import rasterio.mask
import os
from shapely.geometry import shape
#get_ipython().magic('matplotlib inline')


# # Files

# In[6]:


shapefile=gpd.read_file(BAG_Panden)


# In[9]:


### all_ids= []
[all_ids.append(i) for i in shapefile['Identifica']]
print('We now ave all shapefile IDs')


# # Lists

# In[79]:


# Old
nonveg_percent_list=[]
nonveg_abs_list=[]
inten_percent_list=[]
inten_abs_list=[]
exten_percent_list=[]
exten_abs_list=[]
trees_percent_list=[]
trees_abs_list=[]
total_abs_list = []
area_per_roof_list = []
mean = []
max_value = []
min_value = []
nasa_classification_list = []
average_vegetation = []
ID_list = []


# In[80]:


# Statistical information

pct_05 = []
pct_10 = []
pct_15 = []
pct_20 = []
pct_25 = []
pct_30 = []
pct_35 = []
pct_40 = []
pct_45 = []
pct_50 = []
pct_55 = []
pct_60 = []
pct_65 = []
pct_70 = []
pct_75 = []
pct_80 = []
pct_85 = []
pct_90 = []
pct_95 = []

pc_1_08 = []
pc_08_06 = []
pc_06_04 = []
pc_04_02 = []
pc_02_00 = []
pc_00_02 = []
pc_02_04 = []
pc_04_06 = []
pc_06_08 = []
pc_08_1 = []

var = []
std = []


# In[81]:


# Clusters

k2_01 = []
k2_02 = []
k4_01 = []
k4_02 = []
k4_03 = []
k4_04 = []

pc_2_01 = []
pc_2_02 = []
pc_4_01 = []
pc_4_02 = []
pc_4_03 = []
pc_4_04 = []

area_2_01 = []
area_2_02 = []
area_4_01 = []
area_4_02 = []
area_4_03 = []
area_4_04 = []


# # Functions

# In[82]:


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return(image)


# In[83]:


def cluster_per_k(imarray, n_colors):
        
    # Load Image and transform to a 2D numpy array.
    w, h= original_shape = tuple(imarray.shape)
    
    d = 1
    image_array = np.reshape(imarray, (w * h, d))
    
    
    
    #print("Fitting model on a small sub-sample of the data")
    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    
    # Get labels for all points
    #print("Predicting color indices on the full image (k-means)")
    t0 = time()
    labels = kmeans.predict(image_array)
    
    
    codebook_random = shuffle(image_array, random_state=0)[:n_colors + 1]
    #print("Predicting color indices on the full image (random)")
    t0 = time()
    labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)
    
    
    
    #plt.title('Quantized image (2 colors, K-Means)')
    return(recreate_image(kmeans.cluster_centers_, labels, w, h))


# In[84]:


def get_table_of_clusters_from_id(roof_id, shp_file = "/home/data/citycentre/BAG_Panden.shp", raster_file = "/home/data/citycentre/lufo_2014.tif"):
    try:
        ID = roof_id
        shapes = fiona.open(shp_file, "r")
        roof_idx = [str(feat['properties']['Identifica']) for feat in shapes].index(roof_id)
        area_per_roof = shapes[roof_idx]['properties']['SHAPE_Area']
        

        feat1 = [shapes[roof_idx]['geometry'] ]
        with rasterio.open('/home/data/processing/CIR_2015_10.tif') as src:
                out_image, out_transform = rasterio.mask.mask(src, feat1,crop=True, nodata=10, all_touched=True)
                out_meta = src.meta.copy()  
            
         
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        
        NIR = np.matrix(out_image[0, :, :]).astype(float)
        NIR[NIR==10] = 90
        RED = np.matrix(out_image[1, :, :]).astype(float)
        
        ndvi = (NIR-RED)/(NIR+RED)

        ndvi_not_nan = ndvi[ndvi != .8]
        ndvi[ndvi == .8] = np.nan 
        
        k4_ndvi = cluster_per_k(np.array(ndvi_not_nan), 5)
        k2_ndvi = cluster_per_k(np.array(ndvi_not_nan), 3)
              
        list_of_4_cluster_values = (np.unique(k4_ndvi.flatten())).tolist()[:4]
        k4_01.append(list_of_4_cluster_values[0])
        k4_02.append(list_of_4_cluster_values[1])
        k4_03.append(list_of_4_cluster_values[2])
        k4_04.append(list_of_4_cluster_values[3])
        
        list_of_2_cluster_values = (np.unique(k2_ndvi.flatten())).tolist()[:2]
        k2_01.append(list_of_2_cluster_values[0])
        k2_02.append(list_of_2_cluster_values[1])
      
        
        total_abs = (((ndvi_not_nan < 0.088)).sum() + ((0.088 < ndvi_not_nan) & (ndvi_not_nan < 0.21)).sum()                     + ((0.21 < ndvi_not_nan) & (ndvi_not_nan < 0.276)).sum() + ((0.276 < ndvi_not_nan) &                      (ndvi_not_nan < 1)).sum())
        
        nonveg_percent = (((ndvi_not_nan < 0.088)).sum())/total_abs
        exten_percent = (((0.088 < ndvi_not_nan) & (ndvi_not_nan < 0.21)).sum())/total_abs
        inten_percent = (((0.21 < ndvi_not_nan) & (ndvi_not_nan < 0.276)).sum())/total_abs     
        trees_percent = (((0.276 < ndvi_not_nan) & (ndvi_not_nan < 1)).sum())/total_abs    
        nonveg_abs = area_per_roof*nonveg_percent
        exten_abs = area_per_roof*exten_percent
        inten_abs = area_per_roof*inten_percent
        trees_abs = area_per_roof*trees_percent
        mean_roof = np.nanmean(ndvi_not_nan)
        max_roof = np.nanmax(ndvi_not_nan)
        min_roof = np.nanmin(ndvi)
        
        nonveg_percent_list.append(nonveg_percent)
        nonveg_abs_list.append(nonveg_abs)
        inten_percent_list.append(inten_percent)
        inten_abs_list.append(inten_abs)
        exten_percent_list.append(exten_percent)
        exten_abs_list.append(exten_abs)
        trees_percent_list.append(trees_percent)
        trees_abs_list.append(trees_abs)        
        total_abs_list.append(total_abs)
        area_per_roof_list.append(area_per_roof)
        ID_list.append(ID)
        
        min_value.append(min_roof)
        pct_05.append(np.percentile(ndvi_not_nan, 5))
        pct_10.append(np.percentile(ndvi_not_nan, 10))
        pct_15.append(np.percentile(ndvi_not_nan, 15))
        pct_20.append(np.percentile(ndvi_not_nan, 20))        
        pct_25.append(np.percentile(ndvi_not_nan, 25))
        pct_30.append(np.percentile(ndvi_not_nan, 30))
        pct_35.append(np.percentile(ndvi_not_nan, 35))
        pct_40.append(np.percentile(ndvi_not_nan, 40))
        pct_45.append(np.percentile(ndvi_not_nan, 45))
        pct_50.append(np.percentile(ndvi_not_nan, 50))
        pct_55.append(np.percentile(ndvi_not_nan, 55))
        pct_60.append(np.percentile(ndvi_not_nan, 60))        
        pct_65.append(np.percentile(ndvi_not_nan, 65))
        pct_70.append(np.percentile(ndvi_not_nan, 70))
        pct_75.append(np.percentile(ndvi_not_nan, 75))
        pct_80.append(np.percentile(ndvi_not_nan, 80))
        pct_85.append(np.percentile(ndvi_not_nan, 85))
        pct_90.append(np.percentile(ndvi_not_nan, 90))
        pct_95.append(np.percentile(ndvi_not_nan, 95))
        max_value.append(max_roof)
        
        pc_1_08.append((((-1 < ndvi_not_nan) & (ndvi_not_nan < -0.8)).sum())/total_abs) 
        pc_08_06.append((((-0.8 < ndvi_not_nan) & (ndvi_not_nan < -0.6)).sum())/total_abs)
        pc_06_04.append((((-0.6 < ndvi_not_nan) & (ndvi_not_nan < -0.4)).sum())/total_abs)
        pc_04_02.append((((-0.4 < ndvi_not_nan) & (ndvi_not_nan < -0.2)).sum())/total_abs)
        pc_02_00.append((((-0.2 < ndvi_not_nan) & (ndvi_not_nan < 0)).sum())/total_abs)
        pc_00_02.append((((0.0 < ndvi_not_nan) & (ndvi_not_nan < 0.2)).sum())/total_abs)
        pc_02_04.append((((0.2 < ndvi_not_nan) & (ndvi_not_nan < 0.4)).sum())/total_abs)
        pc_04_06.append((((0.4 < ndvi_not_nan) & (ndvi_not_nan < 0.6)).sum())/total_abs)
        pc_06_08.append((((0.6 < ndvi_not_nan) & (ndvi_not_nan < 0.8)).sum())/total_abs)
        pc_08_1.append((((0.8 < ndvi_not_nan) & (ndvi_not_nan < 1)).sum())/total_abs)
        
        pc_2_01.append(((k2_ndvi == list_of_2_cluster_values[0]).sum())/total_abs)
        pc_2_02.append(((k2_ndvi == list_of_2_cluster_values[1]).sum())/total_abs)
        pc_4_01.append(((k4_ndvi == list_of_4_cluster_values[0]).sum())/total_abs)
        pc_4_02.append(((k4_ndvi == list_of_4_cluster_values[1]).sum())/total_abs)
        pc_4_03.append(((k4_ndvi == list_of_4_cluster_values[2]).sum())/total_abs)
        pc_4_04.append(((k4_ndvi == list_of_4_cluster_values[3]).sum())/total_abs)
     
        area_2_01.append((((k2_ndvi == list_of_2_cluster_values[0]).sum())/total_abs)*area_per_roof)
        area_2_02.append((((k2_ndvi == list_of_2_cluster_values[1]).sum())/total_abs)*area_per_roof)
        area_4_01.append((((k4_ndvi == list_of_4_cluster_values[0]).sum())/total_abs)*area_per_roof)
        area_4_02.append((((k4_ndvi == list_of_4_cluster_values[1]).sum())/total_abs)*area_per_roof)
        area_4_03.append((((k4_ndvi == list_of_4_cluster_values[2]).sum())/total_abs)*area_per_roof)
        area_4_04.append((((k4_ndvi == list_of_4_cluster_values[3]).sum())/total_abs)*area_per_roof)
  
    
        std.append(np.std(ndvi_not_nan))
        var.append(np.var(ndvi_not_nan))
        mean.append(np.nanmean(ndvi_not_nan))
    
    except ValueError:
        print('A building partly outside the AoI was just detected.')


# # Code

# In[85]:


[get_table_of_clusters_from_id(x) for x in all_ids]
print('The wait is over.')


# # Dataframe

# In[86]:


raw_df = pd.DataFrame({'ID': pd.Series(ID_list),                       'nasa_nonveg_area': pd.Series(nonveg_abs_list),                        'nasa_nonveg_pc': pd.Series(nonveg_percent_list),                        'nasa_ext_area': pd.Series(exten_abs_list),                       'nasa_ext_pc': pd.Series(exten_percent_list),                       'nasa_int_area': pd.Series(inten_abs_list),                       'nasa_int_pc': pd.Series(inten_percent_list),                       'nasa_tree_area': pd.Series(trees_abs_list),                       'nasa_tree_pc': pd.Series(trees_percent_list),                       'total_area': pd.Series(area_per_roof_list),                       'k2_01': pd.Series(k2_01),                       'k2_02': pd.Series(k2_02),                       'k4_01': pd.Series(k4_01),                       'k4_02': pd.Series(k4_02),                       'k4_03': pd.Series(k4_03),                       'k4_04': pd.Series(k4_04),                       'std': pd.Series(std),                       'var': pd.Series(var),                       'mean': pd.Series(mean),                       'min': pd.Series(min_value),                       'pct_05': pd.Series(pct_05),                       'pct_10': pd.Series(pct_10),                       'pct_15': pd.Series(pct_15),                       'pct_20': pd.Series(pct_20),                       'pct_25': pd.Series(pct_25),                       'pct_30': pd.Series(pct_30),                       'pct_35': pd.Series(pct_35),                       'pct_40': pd.Series(pct_40),                       'pct_45': pd.Series(pct_45),                       'pct_50': pd.Series(pct_50),                       'pct_55': pd.Series(pct_55),                       'pct_60': pd.Series(pct_60),                       'pct_65': pd.Series(pct_65),                       'pct_70': pd.Series(pct_70),                       'pct_75': pd.Series(pct_75),                       'pct_80': pd.Series(pct_80),                       'pct_85': pd.Series(pct_85),                       'pct_90': pd.Series(pct_90),                       'pct_95': pd.Series(pct_95),                       'pc_2_01': pd.Series(pc_2_01),                       'pc_2_02': pd.Series(pc_2_02),                       'pc_4_01': pd.Series(pc_4_01),                       'pc_4_02': pd.Series(pc_4_02),                       'pc_4_03': pd.Series(pc_4_03),                       'pc_4_04': pd.Series(pc_4_04),                       'area_2_01': pd.Series(area_2_01),                       'area_2_02': pd.Series(area_2_02),                       'area_4_01': pd.Series(area_4_01),                       'area_4_02': pd.Series(area_4_02),                       'area_4_03': pd.Series(area_4_03),                       'area_4_04': pd.Series(area_4_04),                       'max': pd.Series(max_value),})


raw_df['nasa_max_pc'] = raw_df[['nasa_nonveg_pc','nasa_ext_pc','nasa_int_pc', 'nasa_tree_pc']].max(axis=1)
raw_df.loc[(raw_df['nasa_nonveg_pc'] == raw_df['nasa_max_pc'], 'naive_label')] = '1'
raw_df.loc[(raw_df['nasa_ext_pc'] == raw_df['nasa_max_pc'], 'naive_label')] = '2'
raw_df.loc[(raw_df['nasa_int_pc'] == raw_df['nasa_max_pc'], 'naive_label')] = '3'
raw_df.loc[(raw_df['nasa_tree_pc'] == raw_df['nasa_max_pc'], 'naive_label')] = '4'


raw_df.set_index('ID', inplace=True)

raw_df = raw_df[['nasa_nonveg_area', 'nasa_nonveg_pc', 'nasa_ext_area', 'nasa_ext_pc',        'nasa_int_area', 'nasa_int_pc', 'nasa_tree_area', 'nasa_tree_pc', 'total_area', 'naive_label',         'k2_01', 'k2_01', 'k4_01', 'k4_02', 'k4_03', 'k4_04', 'std', 'var', 'mean',         'min', 'pct_05', 'pct_10', 'pct_15', 'pct_20', 'pct_25', 'pct_30', 'pct_35', 'pct_40', 'pct_45',         'pct_50', 'pct_55', 'pct_60', 'pct_65', 'pct_70','pct_75', 'pct_80', 'pct_85', 'pct_90', 'pct_95',        'pc_2_01', 'pc_2_02', 'pc_4_01', 'pc_4_02', 'pc_4_03', 'pc_4_04', 'area_2_01', 'area_2_02',        'area_4_01', 'area_4_02', 'area_4_03', 'area_4_04', 'max']].dropna()


# # Output

# In[87]:


raw_df.to_csv('2_processing/features_table.csv')

