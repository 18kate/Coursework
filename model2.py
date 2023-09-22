#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import re
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


# In[3]:


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


# In[4]:


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


# In[5]:


prices['Цена'].plot()


# In[6]:


idx = pd.date_range(prices.index[-1], prices.index[0])
prices = prices.reindex(idx, fill_value=None)
prices['Объём'] = prices['Объём'].fillna(0).astype(int)
prices[['Цена', 'Откр.', 'Макс.', 'Мин.']] = prices[['Цена', 'Откр.', 'Макс.', 'Мин.']].apply(lambda x: x.fillna(prices['Цена'].shift())).ffill()
prices['Биржа работает'] = (prices['Объём'] != 0).astype(int)
prices['Средняя потраченная сумма на выплаты'] = prices.apply(lambda x: res[(res['Начало выплат'] <= x.name)&(res['Конец выплат'] >= x.name)]['Цена за одну бумагу'].sum(), axis=1)
prices


# In[7]:


prices['Биржа работает завтра'] = prices['Биржа работает'].shift(-1)
prices['Предсказание цены'] = prices['Цена'].shift(-1)
prices = prices.dropna()
prices


# In[9]:


res_other = pd.read_csv('sept.csv')
res_other = res_other[res_other['short_company_name'] == 'ПАО Сбербанк'][cols]
res_other = res_other[res_other['Date'] != '0']
res_other['Date'] = pd.to_datetime(res_other['Date'], format='%d.%m.%Y')
res_other = res_other[res_other['doc_type'] != 'Выплаченные доходы или иные выплаты, причитающиеся владельцам ценных бумаг эмитента']
res_other = res_other.groupby('Date').apply(lambda x: True)
res_other.name = 'Было совещание'
res_other


# In[10]:


prices = prices.merge(res_other, left_index=True, right_index=True, how='left')
prices = prices.fillna(0)
prices['Биржа работает завтра'] = prices['Биржа работает завтра'].astype(int)
prices


# In[16]:


class PricesDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return torch.FloatTensor([self.X.iloc[index]]), torch.FloatTensor([self.y.iloc[index]])


X = prices.drop(columns=['Предсказание цены'])
y = prices[['Предсказание цены']]

dataset = PricesDataset(X, y)

train_dataset = PricesDataset(X.loc[:'2023-08-01'], y.loc[:'2023-08-01'])
test_dataset = PricesDataset(X.loc['2023-08-02':], y.loc['2023-08-02':])


# In[17]:


train_loader = DataLoader(train_dataset, shuffle=False, batch_size=15)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=len(test_dataset))


# In[19]:


#Попробовать GRU
#nn.GRU
#Только h0 нужен
class PricesNeuralNet(nn.Module):
    
    def __init__(self, in_features, hidden, num_layers, out_features=1):
        super(PricesNeuralNet, self).__init__()
        
        self.hidden = hidden
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(in_features, hidden, num_layers, batch_first=True)
        
        self.fc1 = nn.Linear(hidden, hidden//2)
        '''
        Попробовать тут функции активации
        '''
        self.fc2 = nn.Linear(hidden//2, out_features)
    
    def forward(self, x):
        
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden).requires_grad_()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden).requires_grad_()
        
        out, (h, c) = self.lstm(x, (h0.detach(), c0.detach()))
        
        out = self.fc1(out)
        return x[:, :, 0, None] + self.fc2(out)
    
#     def predict_price(self, x):
#         pred = self.forward(x)
#         return x[:, :, 0] + pred[:, :, 0]

model = PricesNeuralNet(in_features=dataset.X.shape[1], hidden=50, num_layers=3, out_features=1)


# In[20]:


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# In[23]:


from tqdm import tqdm

epoch = 20
losses = []
mean_losses = []
for e in tqdm(range(epoch)):
    
    model.train()
    mean_loss = 0
    for index, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
#         print(X)
        predict = model(X)
#         print(predict)
        loss = criterion(predict, y)
        loss.backward()
        
        losses.append(loss.item())
        mean_loss += losses[-1]
        optimizer.step()
    
    mean_losses.append(mean_loss/(len(train_loader)))
    
    model.eval()
    test_losses = []
    for index, (X, y) in enumerate(test_loader):
        
        predict = model(X)
        loss = criterion(predict, y)
        
        test_losses.append(loss.item())
    
        
    print(f'Epoch {e} | Mean Loss train - {mean_losses[-1]} | Mean Loss Val {sum(test_losses)}')


# In[28]:


import matplotlib.pyplot as plt

plt.plot(mean_losses)


# In[29]:


for index, (X, y) in enumerate(test_loader):
    predict = model(X)


# In[30]:


predict


# In[31]:


plt.plot(predict[:, :, 0].detach(), c='r')
plt.plot(y[:, :, 0])


# In[ ]:




