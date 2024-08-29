from sklearn.decomposition import PCA
import torch
import os
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
    data_list=[group_prior, group_mlm, group_nomlm]
    labels=[l1] * len(group_prior) + [l2] * len(group_mlm) + [l3] * len(group_nomlm)
    if args.other1_path and args.other2_path:
        l4=args.other1_path.split('/')[-2]
        data_list.append(group_other1)
        labels+= [l4] * len(group_other1)
    if args.other1_path and args.other2_path:
        l5=args.other2_path.split('/')[-2]
        data_list.append(group_other2)
        labels+= [l5] * len(group_other2)
    data = np.vstack(data_list)
    labels=np.array(labels)
    print(labels.shape,'labels.shape')
    print(data.shape,'data.shape')
    # Create labels for the groups

    # Perform t-SNE
    for _ in range(10):
        if args.rs is None:
            rs=np.random.randint(0,9999)
        else:
            rs=args.rs
        if args.reducer=='pca':
            pca = PCA(n_components=2)
            results = pca.fit_transform(data)
        elif args.reducer=='tsne':
            tsne = TSNE(n_components=2,random_state=rs,perplexity=args.perplexity)  # Adjust perplexity if needed
            results = tsne.fit_transform(data)
        else:
            assert False
        # tsne = TSNE(n_components=2,learning_rate='auto',init='random', perplexity=2)  # Adjust perplexity if needed
        
        # print(results.shape,'tsne_results.shape')
        # print(results[labels == l1].shape,"results[labels == l1].shape")
        # print(results[labels == l2].shape,"results[labels == l2].shape")
        # print(results[labels == l3].shape,"results[labels == l3].shape")
        # print(tsne_results[labels == l4].shape,"tsne_results[labels == l4].shape")
        # print(tsne_results[labels == l5].shape,"tsne_results[labels == l5].shape")
        # Plot the t-SNE results
        plt.figure(figsize=(10, 5))
        mid1=[np.mean(results[labels == l1, 0]),np.mean(results[labels == l1, 1])]
        mid2=[np.mean(results[labels == l2, 0]),np.mean(results[labels == l2, 1])]
        mid3=[np.mean(results[labels == l3, 0]),np.mean(results[labels == l3, 1])]
        print(l1,l2,l3,'labels')
        print(mid1,'mid1')
        print(mid2,'mid2')
        print(mid3,'mid3')
        
        plt.scatter(results[labels == l1, 0], results[labels == l1, 1], label='Group {}'.format(l1), alpha=0.6,s=10)
        plt.scatter(results[labels == l2, 0], results[labels == l2, 1], label='Group {}'.format(l2), alpha=0.6,marker='o',s=10)
        plt.scatter(results[labels == l3, 0], results[labels == l3, 1], label='Group {}'.format(l3), alpha=0.6,s=10)
        if args.other1_path:
            plt.scatter(results[labels == l4, 0], results[labels == l4, 1], label='Group {}'.format(l4), alpha=0.6,s=10)
        if args.other2_path:
            plt.scatter(results[labels == l5, 0], results[labels == l5, 1], label='Group {}'.format(l5), alpha=0.6,s=10)
        plt.scatter([mid1[0]],[mid1[1]], marker='x',alpha=0.6,s=30,color='black')
        plt.scatter([mid2[0]],[mid2[1]], marker='x',alpha=0.6,s=30,color='black')
        plt.scatter([mid3[0]],[mid3[1]], marker='x',alpha=0.6,s=30,color='black')
        plt.legend()

        plt.title("Plot Results")
        # plt.xlabel("t-SNE Component 1")
        # plt.ylabel("t-SNE Component 2")
        rand_num=np.random.randint(10000,99999)
        plt.savefig(os.path.join(args.dst_dir,'plot_{}_pp{}_{:06d}_rs{}.jpg'.format(args.reducer,args.perplexity,rand_num,rs)))
        del results
        del tsne
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
    parser.add_argument('--rs',type=int)
    args=parser.parse_args()
    main(args)