import pandas as pd
import statistics
from pandas import DataFrame, read_csv

file = r'sonar_hw1.csv'
df = pd.read_csv(file)
#print(df.head())
#print('Max', df['Highscore'].max())
#print('Min', df['Highscore'].min())


#print("Max R1", df['R1']).max()

r1 = []
r1 = df['R1']
output = []
count = 0

for column in df:
    temp = df[column]
    count += 1
    s = "R"+ str(count)
    print(s,  " -- Max ", round(max(temp),3), "Min ", round(min(temp),3), " Median ", round(statistics.median(temp),3))
    for i in temp:
        if i < 0 or i > 1:
            print(i)
    print("----------------------------------------------")




print("Count is ", count)




#print(output)
