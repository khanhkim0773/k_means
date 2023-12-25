from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

#Đọc dữ liệu từ file Mall_Customers.csv
df = pd.read_csv("./Mall_Customers.csv") 
data=df.head(200)
print(data)

#Elbow
max_k = 10
wcss = []  # Within-Cluster-Sum-of-Squares
for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data[['Age','Annual Income (k$)']])
    wcss.append(kmeans.inertia_)

plt.plot(range(1, max_k + 1), wcss, marker='o')
plt.title('Phương pháp chọn k bằng elbow')
plt.xlabel('Số lượng cụm (k)')
plt.ylabel('WCSS (Within-Cluster-Sum-of-Squares)')
plt.show()

#Kmean

# print(data.shape)
data.info()

# plt.scatter(data['Age'],data['Annual Income (k$)'])
# plt.show()

print(data['Age'].max())
print(data['Age'].min())
# print(data['Annual Income (k$)'].max())
# print(data['Annual Income (k$)'].min())

scaler =  MinMaxScaler()
data['Annual Income (k$)'] = scaler.fit_transform(data[['Annual Income (k$)']])
data['Age'] = scaler.fit_transform(data[['Age']])
print(data.head())

km = KMeans(n_clusters=3)
y_predict=km.fit_predict(data[['Age','Annual Income (k$)']])
#print(y_predict)

data['cluster'] = y_predict
# print(data.head(10))

d1 = data[data.cluster==0]
d2 = data[data.cluster==1]
d3 = data[data.cluster==2]

# plt.scatter(d1['Age'],d1['Annual Income (k$)'])
# plt.show()

# plt.scatter(d2['Age'],d2['Annual Income (k$)'])
# plt.show()

# plt.scatter(d3['Age'],d3['Annual Income (k$)'])
# plt.show()


plt.scatter(d1['Age'],d1['Annual Income (k$)'],color='blue', label = 'Cụm 1')
plt.scatter(d2['Age'],d2['Annual Income (k$)'],color='green', label = 'Cụm 2')
plt.scatter(d3['Age'],d3['Annual Income (k$)'],color='red', label = 'Cụm 3')


# print(km.cluster_centers_)
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color ='black',marker ='*',label = 'Tâm cụm')
plt.xlabel('Tuổi')
plt.ylabel('Thu nhập hằng năm (k$)')
plt.legend()
plt.show()


