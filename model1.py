#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import re
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


# In[20]:


prices = pd.read_csv('ПАО Сбербанк.csv')
prices['Дата'] = pd.to_datetime(prices['Дата'], format='%d.%m.%Y')
prices['Цена'] = prices['Цена'].str.replace(',','.').astype(float)
prices['Откр.'] = prices['Откр.'].str.replace(',','.').astype(float)
prices['Макс.'] = prices['Макс.'].str.replace(',','.').astype(float)
prices['Мин.'] = prices['Мин.'].str.replace(',','.').astype(float)
prices['Объём'] = prices['Объём'].str.replace(',','.').apply(lambda x: int(float(x[:-1]) * 1_000_000) if 'M' in x else int(float(x[:-1]) * 1_000_000_000))
prices = prices.set_index('Дата')
prices = prices.drop(columns=['Изм. %'])
prices


# In[21]:


cols = ['doc_type', 'Отчетный период, за который выплачивались доходы начало',
       'Отчетный период, за который выплачивались доходы конец',
       'Размер выплаченных в расчете на одну ценную бумагу',
       'Количество бумаг по которым выплатили', 'Date']

res = pd.read_csv('sept.csv')
res = res[res['short_company_name'] == 'ПАО Сбербанк'][cols]
res = res[res['Date'] != '0']
res = res.dropna()
res = res[res['doc_type'] == 'Выплаченные доходы или иные выплаты, причитающиеся владельцам ценных бумаг эмитента']
res = res[res['Отчетный период, за который выплачивались доходы конец'].apply(len) == 10]
res['Date'] = pd.to_datetime(res['Date'], format='%d.%m.%Y')
res['Отчетный период, за который выплачивались доходы начало'] = pd.to_datetime(res['Отчетный период, за который выплачивались доходы начало'], format='%d.%m.%Y')
res['Отчетный период, за который выплачивались доходы конец'] = pd.to_datetime(res['Отчетный период, за который выплачивались доходы конец'], dayfirst=True)
res['Размер выплаченных в расчете на одну ценную бумагу'] = res['Размер выплаченных в расчете на одну ценную бумагу'].str.replace(' ', '').apply(lambda x: float(re.findall(r'(?!^)\d+(?:,|\.)\d+', x)[0].replace(',', '.')))
res['Количество бумаг по которым выплатили'] = res['Количество бумаг по которым выплатили'].str.replace(' ', '').apply(lambda x: int(re.findall(r'(\d+)шт\.', x)[0]))
res = res.drop(columns=['doc_type'])
res = res.sort_values('Date').set_index('Date')
res.columns = ['Начало выплат', 'Конец выплат', 'Цена за одну бумагу', 'Кол-во бумаг']
res


# In[22]:


prices['Цена'].plot()


# In[23]:


idx = pd.date_range(prices.index[-1], prices.index[0])
prices = prices.reindex(idx, fill_value=None)
prices['Объём'] = prices['Объём'].fillna(0).astype(int)
prices[['Цена', 'Откр.', 'Макс.', 'Мин.']] = prices[['Цена', 'Откр.', 'Макс.', 'Мин.']].apply(lambda x: x.fillna(prices['Цена'].shift())).ffill()
prices['Биржа работает'] = (prices['Объём'] != 0).astype(int)
prices['Средняя потраченная сумма на выплаты'] = prices.apply(lambda x: res[(res['Начало выплат'] <= x.name)&(res['Конец выплат'] >= x.name)]['Цена за одну бумагу'].sum(), axis=1)
prices


# In[24]:


prices['Биржа работает завтра'] = prices['Биржа работает'].shift(-1)
prices['Предсказание цены'] = prices['Цена'].shift(-1) - prices['Цена']
prices = prices.dropna()
prices


# In[25]:


res_other = pd.read_csv('sept.csv')
res_other = res_other[res_other['short_company_name'] == 'ПАО Сбербанк'][cols]
res_other = res_other[res_other['Date'] != '0']
res_other['Date'] = pd.to_datetime(res_other['Date'], format='%d.%m.%Y')
res_other = res_other[res_other['doc_type'] != 'Выплаченные доходы или иные выплаты, причитающиеся владельцам ценных бумаг эмитента']
res_other = res_other.groupby('Date').apply(lambda x: True)
res_other.name = 'Было совещание'
res_other


# In[26]:


prices = prices.merge(res_other, left_index=True, right_index=True, how='left')
prices = prices.fillna(0)
prices['Биржа работает завтра'] = prices['Биржа работает завтра'].astype(int)
prices


# In[27]:


class PricesDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return torch.FloatTensor(self.X.iloc[index]), torch.FloatTensor(self.y.iloc[index])


X = prices.drop(columns=['Предсказание цены'])
y = prices[['Предсказание цены']]

dataset = PricesDataset(X, y)
dataset[0]


# In[28]:


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# In[29]:


class PricesNeuralNet(nn.Module):
    
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.in_features = in_features
#         self.n_layers = n_layers
        self.out_features = out_features
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Sigmoid()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features, out_features)
        )
    
    def forward(self, x):
        x = self.fc1(x)
        return self.fc4(x)

model = PricesNeuralNet(in_features=dataset.X.shape[1])


# In[30]:


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


# In[32]:


from tqdm import tqdm

epoch = 20
losses = []
mean_losses = []
for e in tqdm(range(epoch)):
    
    model.train()
    mean_loss = 0
    for index, (X, y) in enumerate(train_dataset):
        
        predict = model(X)
        loss = criterion(predict, y)
        loss.backward()
        
        losses.append(loss.item())
        mean_loss += losses[-1]
        optimizer.step()
    
    scheduler.step()
    mean_losses.append(mean_loss/len(train_dataset))
    
    model.eval()
    test_losses = []
    for index, (X, y) in enumerate(test_dataset):
        
        predict = model(X)
        loss = criterion(predict, y)
        
        test_losses.append(loss.item())
    
        
    print(f'Epoch {e} | Mean Loss train - {mean_losses[-1]} | Mean Loss Val {sum(test_losses)/len(test_dataset)}')


# In[33]:


import matplotlib.pyplot as plt

plt.plot(mean_losses)


# In[35]:


model(train_dataset[40][0])


# In[ ]:




