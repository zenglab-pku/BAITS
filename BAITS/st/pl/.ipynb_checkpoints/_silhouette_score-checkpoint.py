import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def silhouette_scores(model, k_range: tuple, figsize=(6, 4)):
    """
    Plot silhouette scores for a given range of clusters.
    
    Parameters
    ----------
    model: object
        The clustering model that has an attribute `silhouette_scores` (list of lists with silhouette scores).
        
    k_range: tuple
        A tuple (n, m) representing the range of number of clusters to evaluate.
        
    figsize: tuple, optional
        The size of the plot (default is (6, 4)).
    """
    # Extract silhouette scores into a DataFrame
    df = pd.DataFrame(
        model.silhouette_scores, 
        columns=list(range(*(max(2, k_range[0] - 1), k_range[1] + 1)))  # k_range[0] is the lower bound, k_range[1] is the upper bound
    )

    # Calculate mean and standard deviation of the silhouette scores
    mean_scores = df.mean(axis=0)
    std_scores = df.std(axis=0)

    # Plotting the silhouette scores
    plt.figure(figsize=figsize)
    sns.lineplot(x=mean_scores.index, y=mean_scores, marker='o', color='cornflowerblue', label='Mean silhouette score')
    plt.fill_between(mean_scores.index, mean_scores - std_scores, mean_scores + std_scores, 
                     alpha=0.3, color='gray', label='Variance')

    # Customize plot
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.legend()
    plt.show()
