import pandas as pd

file = r'sonar_hw1.csv'
df = pd.read_csv(file)

# removing the last column with 'R' and 'M'
df_new = df.iloc[0:, 0:60]

#defining variables
output = []
count = 1
s = []
n_low = []
n_high = []
total2 = []
n_null_total = []
total = []

#creating a list with the row names
for i in range(1, 61):
    s.append("R" + str(i))


#iterating the dataframe columnwise
for column in df_new:
    temp = df_new[column]
    for i in temp:
        #removing teh outliers
        if i > 0 and i < 1 :

            temp2 = []
            count += 1
            temp2.append(max(temp))
            temp2.append(min(temp))
        if count < 61:
            temp2.append(df['R'+ str(count)].median())
            total.append(temp2)

#iterting the dataframe, without removing the outliers
for column in df_new:
    temp = df_new[column]
    n_low = []
    n_high = []
    n_null = []
    count = 0
    temp2 = []
    for i in temp:
        if i < 0:
            n_low.append(i)
        elif i > 1:
            n_high.append(i)
        elif i >= 0 and i <= 1:
            continue
        else:
            count += 1
    n_null.append(count)



    temp2.append(len(n_low))
    temp2.append(len(n_high))
    temp2.append(count)
    total2.append(temp2)

row_max = []
row_min = []
row_median = []
row_low = []
row_high = []
row_null = []

for i in total:
    row_max.append(i[0])
    row_min.append(i[1])
    row_median.append(i[2])
for i in total2:
    row_low.append(i[0])
    row_high.append(i[1])
    row_null.append(i[2])

#creating an output dataframe with the different lists
df_out = pd.DataFrame({ "Missing": row_null, "n_low": row_low, "n_high": row_high,"Median": row_median,"Min": row_min,  "Max":row_max})
df_out.insert(loc = 0,column = "Row", value = s)
print(df_out)






