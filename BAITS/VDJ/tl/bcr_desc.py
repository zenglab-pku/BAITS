import numpy as np
import pandas as pd

def renyi_entropy(probabilities, alpha_values=range(10)):
    """
    Calculate the Rényi entropy for a given probability distribution and alpha.
    
    Parameters:
    probabilities (list or numpy array): Probability distribution (should sum to 1).
    alpha (float): The order of the Rényi entropy.
    
    Returns:
    float: The calculated Rényi entropy.
    """
    results = {}
    for alpha in alpha_values:
        if alpha == 1:
            entropy = -np.sum(probabilities * np.log(probabilities))
        else:
            entropy = 1 / (1 - alpha) * np.log(np.sum(probabilities**alpha))
        results[alpha] = [entropy]
    return pd.DataFrame.from_dict(results)


def shannon_entropy(p): 
    p = p[p > 0]  # Only consider non-zero probabilities
    H = -np.sum(p * np.log2(p))
    return H


def CPK(count): 
    return len(count)/sum(count) * 1000

    
def normalize_shannon_entropy(p): 
    p = p[p > 0]  # Only consider non-zero probabilities
    H = -np.sum(p * np.log2(p)) / np.log2(len(p))
    return H

def gini_index(data): 
    if len(data) == 0:
        raise ValueError("Input data cannot be empty.")
    sorted_data = np.sort(data)
    n = len(data)
    cumulative_sum = np.cumsum(sorted_data)
    gini_index = (n + 1 - 2 * np.sum(cumulative_sum) / np.sum(sorted_data)) / n
    return gini_index

def Clonality(p):
    C = 1 - shannon_entropy(p) / np.log2(len(p))
    return C

def compute_index(function_name, p):
    # 使用 globals() 来根据输入的函数名称调用对应的函数 
    functions = {
        'shannon_entropy': shannon_entropy,
        'normalize_shannon_entropy': normalize_shannon_entropy,
        'Clonality': Clonality, 
        'renyi_entropy': renyi_entropy,
        'gini_index': gini_index,
        'CPK':CPK
    }
    
    # 检查用户输入的函数是否在已定义的函数字典中
    if function_name in functions:
        if function_name=='renyi_entropy':
            return functions[function_name](p)
        else:
            return functions[function_name](p)
    else:
        raise ValueError(f"Function '{function_name}' not recognized.")
        