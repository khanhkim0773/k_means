
# Import các module cần thiết
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Đọc dữ liệu từ file Mall_Customers.csv
df = pd.read_csv("./Mall_Customers.csv") 
data = df.head(200)

# Chuẩn hóa dữ liệu bằng MinMaxScaler
scaler = MinMaxScaler()
data['Annual Income (k$)'] = scaler.fit_transform(data[['Annual Income (k$)']])
data['Age'] = scaler.fit_transform(data[['Age']])
data['Spending Score'] = scaler.fit_transform(data[['Spending Score (1-100)']])


# Định nghĩa hàm kmeans
def kmeans(features, k, max_iterations=600, tol=1e-4):
    # Khởi tạo các tâm cụm ngẫu nhiên
    initial_centers = features[np.random.choice(features.shape[0], k, replace=False)]
    
    for _ in range(max_iterations):

        #Tạo mảng 2 chiều lưu khoảng cách giữa tâm cụm đến các điểm dữ liệu
        distances = np.zeros((initial_centers.shape[0], features.shape[0]))
        for i in range(initial_centers.shape[0]):
            for j in range(features.shape[0]):
                diff = features[j] - initial_centers[i]
                distance = np.sqrt(np.sum(diff ** 2)) #Tính khoảng cách euclid
                distances[i, j] = distance

        # Gán nhãn dựa trên tâm cụm gần nhất
        #tạo mảng labels có số lượng pt bằng số cột của mảng distances
        labels = np.zeros(distances.shape[1], dtype=int) 
        for j in range(distances.shape[1]):
            min_distance = float('inf')
            min_index = -1
            for i in range(distances.shape[0]):
                if distances[i, j] < min_distance:
                    min_distance = distances[i, j]
                    min_index = i
            labels[j] = min_index

        
        old_centers = np.copy(initial_centers) # Lưu trữ tâm cụm hiện tại
        #Cập nhật tâm cụm dựa trên trung bình các điểm thuộc cùng một cụm
        for i in range(k):
            if np.sum(labels == i) > 0:
                count = np.sum(labels == i)
                center_sum = np.sum(features[labels == i], axis=0)
                initial_centers[i] = center_sum / count

        # Kiểm tra xem nếu tâm cụm không đổi thì kết thúc vòng lặp
        if np.allclose(old_centers, initial_centers, atol=tol):
            break

    return labels, initial_centers


#Định nghĩa hàm elbow_method
def elbow_method(features, max_k):
    wcss = []  # Mảng lưu tổng bình phương khoảng cách trong cụm
    for k in range(1, max_k + 1):
        labels, _ = kmeans(features, k)
        # Tính tổng bình phương khoảng cách trong cụm (WCSS) và thêm vào danh sách
        wcss.append(np.sum((features - _[labels]) ** 2))
    return wcss

max_k = 10
wcss_result = elbow_method(data[['Age', 'Annual Income (k$)']].values, max_k)

# Hiển thị biểu đồ elbow
plt.plot(range(1, max_k + 1), wcss_result, marker='o')
plt.title('Phương pháp chọn k bằng elbow')
plt.xlabel('Số lượng cụm (k)')
plt.ylabel('WCSS (Tổng bình phương khoảng cách trong cụm)')
plt.show()

# ##Áp dụng KMeans clustering

k_optimal = 3 #đặt số k tối ưu
colors = ['blue', 'green', 'black', 'cyan', 'yellow', 'magenta']
###################################################################################################
##Phân cụm trên không gian 2D

# labels, centers = kmeans(data[['Age', 'Annual Income (k$)']].values, k_optimal)

# ##Thêm nhãn cụm vào dataframe
# data['cluster'] = labels
# ##print(data.head(10))

# clusters = []
# for i in range(k_optimal):
#     cluster_data = data[data.cluster == i]
#     clusters.append(cluster_data)
# print(clusters)



# labels = [f'Cụm {i+1}' for i in range(k_optimal)]

# for i, cluster in enumerate(clusters):
#     plt.scatter(cluster['Age'], cluster['Annual Income (k$)'], color=colors[i], label=labels[i])

# ## Hiển thị trung tâm cụm
# plt.scatter(centers[:, 0], centers[:, 1], color='red', marker='*', label='Tâm cụm')
# plt.xlabel('Tuổi')
# plt.ylabel('Thu nhập hằng năm (k$)')
# plt.legend()
# plt.title(f'KMeans với k={k_optimal}')
# plt.show()

#######################################################################################################3


##########################################################################################################
#Phân cụm trên cơ sở dữ liệu 3d
selected_features = data[['Age', 'Annual Income (k$)', 'Spending Score']]
labels, centers = kmeans(selected_features.values, k_optimal)
data['cluster'] = labels

clusters = []
for i in range(k_optimal):
    cluster_data = data[data.cluster == i]
    clusters.append(cluster_data)
print(clusters)
#Tạo biểu đồ 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Vẽ biểu đồ 3D cho mỗi cụm
for i, cluster in enumerate(clusters):
   ax.scatter(cluster['Age'], cluster['Annual Income (k$)'], cluster['Spending Score'],
              color=colors[i], label=f'Cụm {i + 1}')

# Hiển thị trung tâm cụm
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color='red', marker='*', label='Tâm cụm')

ax.set_xlabel('Tuổi')
ax.set_ylabel('Thu nhập hằng năm (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.legend()
ax.set_title(f'KMeans với k={k_optimal}')

plt.show()
###########################################################################################################