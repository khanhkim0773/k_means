import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Tạo dữ liệu mẫu
n_samples = 300
X, _ = make_blobs(n_samples=n_samples, centers=4, random_state=42)

# Sử dụng phương pháp elbow để chọn số lượng cụm k
wcss = []  # Within-Cluster-Sum-of-Squares
max_k = 10

for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Vẽ biểu đồ elbow
plt.plot(range(1, max_k + 1), wcss, marker='o')
plt.title('Phương pháp chọn k bằng elbow')
plt.xlabel('Số lượng cụm (k)')
plt.ylabel('WCSS (Within-Cluster-Sum-of-Squares)')
plt.show()
