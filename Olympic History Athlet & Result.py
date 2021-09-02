#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# #### Data

# In[2]:


data = pd.read_csv('D:/Semester 6/Data Mining/Tugas 1/athlete_events.csv')
data.tail(n = 20)


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


np.sum(data.isnull())


# ### *Prepocessing*

# ### Merubah Tipe Variabel yang Sesuai dengan Data
# Menurut informasi data yang telah ditunjukkan di atas, terdapat ketidaksesuaian tipe data variabel ID. Variabel ID terdeteksi sebagai integer. Sebenarnya ID merupakan tipe data kategori. Begitu juga dengan variabel tahun.

# In[6]:


data['ID'] = data['ID'].astype(object)
data['Year'] = data['Year'].astype(object)
data.info()


# ### *Data Integration*

# Untuk mengetahui apakah setiap NOC hanya memfasilitasi satu tim maka dilakukan subset antara NOC dan tim untuk mengetahui jumlah tim pada setiap kode NOC.

# In[7]:


print(data.loc[:, ['NOC', 'Team']].drop_duplicates()['NOC'].value_counts().head())


# Dari output di atas dapat diketahui bahwa NOC jenis FRA memfasilitasi 160 tim. Ini terlihat terlalu banyak tim, maka perlu dilakukan integrasi data dengan menggabungkan data master NOC untuk mendapatkan region dari NOC

# In[8]:


regions = pd.read_csv('D:/Semester 6/Data Mining/Tugas 1/noc_regions.csv')


# **Merge Data Region dan Data Utama**

# In[9]:


data = data.merge(regions, left_on = 'NOC', right_on = 'NOC', how = 'left')
data.tail(n = 20)


# In[10]:


## Menghapus Variabel Games
data.drop('Games', axis = 1, inplace = True)
data


# In[11]:


## PLOT UNTUK MENENTUKAN IMPUTASI DENGAN MEAN ATAU MEDIAN
plt.figure(figsize = (11,4))
plt.subplot(131)
sns.boxplot(x = data['Height'], saturation = 1, width = 0.8, color = 'tab:cyan', orient = 'v')
plt.subplots_adjust(wspace = 0.3)
plt.subplot(132)
sns.boxplot(x = data['Weight'], saturation = 1, width = 0.5, color = 'tab:cyan', orient = 'v')
plt.subplot(133)
sns.boxplot(x = data['Age'], saturation = 1, width = 0.8, color = 'tab:cyan', orient = 'v')
plt.show()


# In[12]:


np.sum(data.isnull())


# Ternyata setelah digabungkan terdapat data region yang kosong atau missing value.

# ### *Data Cleaning*
# #### 1. *Missing Value*
# Berdasarkan informasi data di atas, terdeteksi *missing value* pada variabel *age*, *height*, *weight*, dan *medal*. 
# 
#    **A. Variabel Medal**
# 
#    Pada variabel *medal* terdapat 231.333 baris yang kosong, hal ini berarti atlit tersebut tidak memenangkan olimpiade sehingga tidak mendapat mendali. Maka dari itu, baris yang kosong tersebut dapat diisi dengan *No Medal*

# In[13]:


data['Medal'].fillna('No Medal', inplace = True)


# In[14]:


np.sum(data.isnull())


# **B. Variabel Height**
# 
#    Pada variabel *Height* terdapat 60171 baris yang kosong (22% data). Karena nilai pada variabel tinggi badan atlit cukup heterogen, yaitu dapat diamati dari boxplot terdapat outlier dan outlier tersebut tidak simetris, maka digunakan *imputation* terhadap *missing value* dengan menggunakan nilai *mean* dari tinggi badan atlit. Berdasarkan *output* deskripsi data diperoleh nilai rata-rata tinggi badan atlit sebesar 175.338970.

# In[15]:


data['Height'] = data['Height'].fillna(data['Height'].median())


# **B. Variabel Weight**
# 
#    Pada variabel *Weight* terdapat 62875 baris yang kosong (23% data). Karena nilai pada variabel berat badan atlit cukup heterogen, yaitu dapat diamati dari boxplot terdapat outlier dan outlier tersebut tidak simetris, maka digunakan *imputation* terhadap *missing value* dengan menggunakan nilai *mean* dari berat badan atlit. Berdasarkan *output* deskripsi data diperoleh nilai rata-rata berat badan atlit sebesar 70,702393.

# In[16]:


data['Weight'] = data['Weight'].fillna(data['Weight'].median())


# **C. Variabel Age**
# 
#    Pada variabel *Age* terdapat 9.474 baris yang kosong (3% data). Karena nilai pada variabel usia atlit tidak cukup homogen, maka digunakan *imputation* terhadap *missing value* dengan menggunakan nilai *median* dari usia atlit. Berdasarkan *output* deskripsi data diperoleh nilai median usia atlit sebesar 24.

# In[17]:


data['Age'] = data['Age'].fillna(data['Age'].median())


# **D. Variabel region**
# 
#    Pada variabel *region* terdapat 370 baris yang kosong. Karena terdapat variabel NOC yang dapat membantu region yang sesuai maka dimunculkan informasi berikut.

# In[18]:


data.loc[data['region'].isnull(),['NOC', 'Team']].drop_duplicates()


# Dari informasi di atas, maka dapat diisi region tim yang masih kosong sebagai berikut dan variabel region ditiadakan dan diganti dengan tim supaya data tim dapat diseragamkan.

# In[19]:


data['region'] = np.where(data['NOC']=='SGP', 'Singapore', data['region'])
data['region'] = np.where(data['NOC']=='ROT', 'Refugee Olympic Athletes', data['region'])
data['region'] = np.where(data['NOC']=='UNK', 'Unknown', data['region'])
data['region'] = np.where(data['NOC']=='TUV', 'Tuvalu', data['region'])

data.drop('Team', axis = 1, inplace = True)
data.rename(columns = {'region': 'Team'}, inplace = True)
data.tail(n = 20)


# **E. Variabel notes**
# 
#    Pada variabel *notes* terdapat 266077 baris yang kosong. Dimana yang kosong berarti tidak ada catatan. Maka dapat diimputasi dengan 'no note'

# In[20]:


data['notes'].fillna('No Note', inplace = True)


# In[21]:


np.sum(data.isnull())


# Maka, sekarang tidak terdapat *missing value* pada data *120 Years of Olympic History : Athletes and Results*

# #### 2. Deteksi *Outlier*

# In[22]:


plt.figure(figsize = (11,4))
plt.subplot(131)
sns.boxplot(x = data['Height'], saturation = 1, width = 0.8, color = 'tab:cyan', orient = 'v')
plt.subplots_adjust(wspace = 0.3)
plt.subplot(132)
sns.boxplot(x = data['Weight'], saturation = 1, width = 0.5, color = 'tab:cyan', orient = 'v')
plt.subplot(133)
sns.boxplot(x = data['Age'], saturation = 1, width = 0.8, color = 'tab:cyan', orient = 'v')
plt.show()


# Berdasarkan boxplot di atas, terdapat outlier sehingga perlu dilakukan analisis univariat outlier dan multivariat outlier

# ### Karakteristik Data

# In[23]:


data.describe()


# ### Visualisasi Data

# ### 5 Negara Teratas Berdasarkan Jumlah Medali Emas

# In[24]:


gold = data[(data.Medal == 'Gold')]
totalGold = gold.Team.value_counts().reset_index(name='Medal').head(10)

sns.catplot(x="index", y="Medal", data=totalGold,
                height=8, kind="bar", palette='bone')
plt.title('10 Negara Teratas Berdasarkan Jumlah Mendali Emas', fontsize = 15)
plt.xlabel('Countries (Negara)', fontsize = 14)
plt.ylabel('Jumlah Mendali Emas', fontsize = 14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()


# #### Line Chart Jumlah Laki-Laki dan Perempuan Dari Tahun ke Tahun

# In[25]:


athletesMen = data[(data.Sex == 'M')]
athletesWomen = data[(data.Sex == 'F')]

summerMen = athletesMen[(athletesMen.Season == 'Summer')]
winterMen = athletesMen[(athletesMen.Season == 'Winter')]
summerWomen = athletesWomen[(athletesWomen.Season == 'Summer')]
winterWomen = athletesWomen[(athletesWomen.Season == 'Winter')]

summerTicks = list(summerMen['Year'].unique())
summerTicks.sort()
winterTicks = list(winterMen['Year'].unique())
winterTicks.sort()

plt.figure(figsize=(15,12))

plt.subplot(221)
partSummerMen = summerMen.groupby('Year')['Sex'].value_counts()
partSummerMen.loc[:,'M'].plot(linewidth=4, color='b')
plt.xticks(summerTicks, rotation=90)
plt.title('Jumlah Atlit Laki-laki Pada Summer Olympics', fontsize=14)

plt.subplot(222)
partSummerWomen = summerWomen.groupby('Year')['Sex'].value_counts()
partSummerWomen.loc[:,'F'].plot(linewidth=4, color='r')
plt.xticks(summerTicks, rotation=90)
plt.title('Jumlah Atlit Perempuan Pada Summer Olympics', fontsize=14)

plt.subplot(223)
partWinterMen = winterMen.groupby('Year')['Sex'].value_counts()
partWinterMen.loc[:,'M'].plot(linewidth=4, color='b')
plt.xticks(winterTicks, rotation=90)
plt.title('Jumlah Atlit Laki-laki Pada Winter Olympics', fontsize=14)

plt.subplot(224)
partWinterWomen = winterWomen.groupby('Year')['Sex'].value_counts()
partWinterWomen.loc[:,'F'].plot(linewidth=4, color='r')
plt.xticks(winterTicks, rotation=90)
plt.title('Jumlah Atlit Perempuan Pada Winter Olympics', fontsize=14)
plt.show()


# #### Boxplot Usia Atlet Berdasarkan Musim dan Jenis Kelamin

# In[26]:


sns.boxplot(x='Season',y='Age',hue='Sex',data=data, palette = 'Pastel1');
plt.title('Boxplot Usia Atlet Berdasarkan Musim dan Jenis Kelamin', fontsize = 15)
plt.xlabel('Musim (Season)', fontsize = 14)
plt.ylabel('Usia (Age)', fontsize = 14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)


# #### Scatter Plot Berat, Tinggi, dan BMI Atlit

# In[27]:


# Menghitung BMI
met = data[(data["Height"].notnull()) & (data["Weight"].notnull())]
# Konversi satuan Tinggi badan menjadi meter
met["Height_mt"] = met["Height"]/100
# BMI
met["BMI"] = met["Weight"]/(met["Height_mt"]**2)

# SCATTER PLOT
plt.figure(figsize=(13,9))
sns.set_style("whitegrid")
plt.tight_layout() 
plt.scatter(met["Height"] ,met["Weight"], cmap="jet", c=met["BMI"],linewidth=.4,edgecolor = "k")
lab = plt.colorbar()
lab.set_label("Body Mass Index",fontsize = 13)
plt.xlabel("Height (Cm)", fontsize = 14)
plt.ylabel("Weight (Kg)", fontsize = 14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title("Scatter Plot Tinggi Badan, Berat Badan, dan BMI Atlit", fontsize = 16)
plt.show()


# #### Boxplot Usia pada Setiap Kategori Mendali Atlet

# In[28]:


plt.figure(figsize=(7,7))
sns.boxplot(x='Medal',y='Age',data=data, width=0.7);
plt.xlabel('Medal', fontsize = 14)
plt.ylabel('Usia (Age)', fontsize = 14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)


# #### Distribusi Medali Emas Berdasarkan Usia (Histogram)

# In[29]:


plt.figure(figsize=(22, 10))
sns.set_style("whitegrid")
plt.tight_layout() 
sns.countplot(gold['Age'], palette = 'cool')
plt.xlabel('Age', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title('Distribution Mendali Emas Berdasarkan Usia Atlit', fontsize = 16)
plt.show()


# #### Jumlah Mendali Emas yang Diperoleh Atlit Usia >50 Tahun pada Setiap Jenis Olimpiade (Bar Chart)

# In[30]:


gold50 = gold['Sport'][gold['Age'] > 50]
plt.figure(figsize=(10, 5))
plt.tight_layout()
sns.countplot(gold50, palette = 'tab10')
plt.xlabel('Sport', fontsize = 15)
plt.ylabel('Jumlah', fontsize = 15)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.title('Jumlah Mendali Emas Atlit Usia > 50 Tahun Setiap Jenis Olimpiade', fontsize = 16)


# In[207]:


### Mencoba mengatasi outlier tapi masih gagal
from sklearn.preprocessing import StandardScaler
data['Weight'] = StandardScaler().fit(data[['Weight']]).transform(data[['Weight']])


# In[213]:


data.loc[data.Weight<3]


# 
