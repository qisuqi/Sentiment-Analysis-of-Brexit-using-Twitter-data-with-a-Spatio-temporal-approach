import pandas as pd

file1 = pd.read_csv('xxx.csv')

#Clean the data
file1 = file1[file1.text != 'RT']
file1['Time'] = pd.to_datetime(file1.created_at)
file1['user'] = file1['user'].astype(str)

new1 = file1['user'].str.split(',',expand=True)
cols1 = [range(5,56)]
new1 = new1.drop(new1.columns[cols1],axis=1)

new1.columns = ['id','id_str','name','screen_name','location']
new1 = new1.drop(columns=['id'])

ID1 = new1['id_str'].str.split(':',expand=True)
Name1 = new1['name'].str.split(':',expand=True)
Screen_name1 = new1['screen_name'].str.split(':',expand=True)
Location1 = new1['location'].str.split(':',expand=True)

data1 = pd.concat([file1,ID1,Name1,Screen_name1,Location1],axis=1,sort=False)
data1.columns = ['0','created_at','Language','Text','user','Time','1','ID','2','Name','3','4','8','Screen_name','5','Location','6','7']
data1 = data1.drop(columns=['0','1','2','3','4','5','6','7','8','created_at','user'])
data1 = data1[['Time','ID','Name','Screen_name','Text','Language','Location']]

data1.to_csv('xxx.csv')

