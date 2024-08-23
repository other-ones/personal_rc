from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
import torch
import os
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm

def main(args):
    os.makedirs(args.dst_dir,exist_ok=True)
    
    group_prior=torch.load(args.prior_path).detach().cpu().numpy()
    group_mlm=torch.load(args.mlm_path).detach().cpu().numpy()
    group_nomlm=torch.load(args.nomlm_path).detach().cpu().numpy()
    if args.other1_path:
        group_other1=torch.load(args.other1_path).detach().cpu().numpy()
    if args.other1_path:
        group_other2=torch.load(args.other2_path).detach().cpu().numpy()
    print(group_mlm.shape,'group_mlm.shape')
    print(group_prior.shape,'group_prior.shape')
    print(group_nomlm.shape,'group_nomlm.shape')
    # Assuming you have three groups of tensors, each of shape (n_samples, 400768)
    # Replace these with your actual tensors
    # group_prior = np.random.rand(100, 400768)  # Replace with your actual data
    # group_mlm = np.random.rand(100, 400768)  # Replace with your actual data
    # group_nomlm = np.random.rand(100, 400768)  # Replace with your actual data

    # Combine the groups into a single dataset
    l1=args.prior_path.split('/')[-2]
    l2=args.mlm_path.split('/')[-2]
    l3=args.nomlm_path.split('/')[-2]
    if args.other1_path and args.other2_path:
        l4=args.other1_path.split('/')[-2]
        l5=args.other2_path.split('/')[-2]
        data = np.vstack((group_prior, group_mlm, group_nomlm,group_other1,group_other2))
        labels = np.array([l1] * len(group_prior) + [l2] * len(group_mlm) + [l3] * len(group_nomlm)+ [l4] * len(group_other1)+ [l5] * len(group_other2))
    else:
        data = np.vstack((group_prior, group_mlm, group_nomlm))
        labels = np.array([l1] * len(group_prior) + [l2] * len(group_mlm) + [l3] * len(group_nomlm))
    print(data.shape,'data.shape')
    # Create labels for the groups

    # Perform t-SNE
    
    if args.reducer=='pca':
        pca = PCA(n_components=2)
        results = pca.fit_transform(data)
    elif args.reducer=='tsne':
        tsne = TSNE(n_components=2,perplexity=args.perplexity)  # Adjust perplexity if needed
        results = tsne.fit_transform(data)
    else:
        assert False
    # tsne = TSNE(n_components=2,learning_rate='auto',init='random', perplexity=2)  # Adjust perplexity if needed
    kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=42)
    cluster_labels = kmeans.fit_predict(results)
    centroids = kmeans.cluster_centers_
    distances = cdist(centroids, centroids, metric='euclidean')

    # Plotting
    plt.figure(figsize=(10, 5))
    unique_labels = np.unique(labels)
    # colors = cm.get_cmap('tab10', len(unique_labels))
    colors=np.random.randint(0,256,(len(unique_labels),3))


    for i, label in enumerate(unique_labels):
        idx = labels == label
        plt.scatter(results[idx, 0], results[idx, 1], label=label, alpha=0.6, s=10,)
        # Draw contours for each cluster
        kde = gaussian_kde(results[idx].T)
        x, y = np.mgrid[results[:, 0].min():results[:, 0].max():100j, results[:, 1].min():results[:, 1].max():100j]
        positions = np.vstack([x.ravel(), y.ravel()])
        f = np.reshape(kde(positions).T, x.shape)
        plt.contour(x, y, f, alpha=0.5)

    # Draw lines between centroids and denote distances
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            plt.plot([centroids[i, 0], centroids[j, 0]], [centroids[i, 1], centroids[j, 1]], 'k--', linewidth=1)
            mid_point = (centroids[i] + centroids[j]) / 2
            plt.text(mid_point[0], mid_point[1], f'{distances[i, j]:.2f}', fontsize=8, ha='center')

    plt.legend()
    plt.title("t-SNE/PCA Clustering and Distances")
    rand_num=np.random.randint(10000,99999)
    plt.savefig(os.path.join(args.dst_dir,'plot_{}_pp{}_{:06d}.jpg'.format(args.reducer,args.perplexity,rand_num)))
    # plt.show()
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--perplexity',type=int)
    parser.add_argument('--reducer')
    parser.add_argument('--prior_path')
    parser.add_argument('--mlm_path')
    parser.add_argument('--nomlm_path')
    parser.add_argument('--dst_dir')
    parser.add_argument('--other1_path')
    parser.add_argument('--other2_path')
    args=parser.parse_args()
    main(args)