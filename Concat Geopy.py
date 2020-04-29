import pandas as pd

file = pd.read_csv('r.csv')
file1 = pd.read_csv('r1.csv')
file2 = pd.read_csv('r2.csv')
file3 = pd.read_csv('r3.csv')
file4 = pd.read_csv('r4.csv')
file5 = pd.read_csv('r5.csv')
file6 = pd.read_csv('r6.csv')
file7 = pd.read_csv('r7.csv')
file8 = pd.read_csv('r8.csv')
file9 = pd.read_csv('r9.csv')

Result = pd.concat([file,file1,file2,file3,file4,file5,file6,file7,file8,file9]).reset_index()

Result = Result.drop(columns={'index','Unnamed: 0'})

Result.to_csv('Result_after.csv')
