def find_eps(data, min_pts, start_eps, end_eps, step):
        eps_range = np.arange(start_eps, end_eps, step)
        best_eps = []
        

        for eps in eps_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_pts)
            dbscan.fit(data)
            clusters = dbscan.fit_predict(normalized_data)

            # Подсчитываем количество кластеров
            num_clusters = len(np.unique(dbscan.labels_)) - 1

            if num_clusters == 4:
                cluster_counts = np.bincount(clusters+1)
                for cluster, count in enumerate(cluster_counts):
                    if (cluster == 3 and count > 80): 
                        best_eps.append(eps)
                
                

        return best_eps

# Параметры для подбора eps
min_pts = 3
start_eps = 0.1
end_eps = 1.0
step = 0.0001

    # Вызов функции для подбора eps
best_eps = find_eps(normalized_data, min_pts, start_eps, end_eps, step)
print(best_eps)