import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.linear_model

from tqdm import tqdm

def compute_propensity_scores(dataset, group_masks, confounder_cols):
    """Compute the propensity scores of a sample.
    
    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset from which the two groups are drawn.
    group_masks : list of pd.Series
        A list of boolean masks identifying members of each group.
    confounder_cols : list of str
        The labels of the dataset columns corresponding to confounders.
        
    Returns
    -------
    group_data : pd.DataFrame
        The treatment and control groups.
    """
    groups = [dataset[group_mask].copy() for group_mask in group_masks]
    for i, group in enumerate(groups):
        group['treatment'] = i
    group_data = pd.concat(groups)
    X_train, y_train = group_data[confounder_cols], group_data['treatment']
    
    propensity_model = sklearn.linear_model.LogisticRegression().fit(X_train, y_train)
    group_data['propensity_scores'] = list(propensity_model.predict_proba(X_train))
    
    return group_data

def psm(dataset, group_masks, confounder_cols, max_caliper=0.2):
    """Perform propensity score caliper matching between the treatment and control groups.
    
    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset from which the two groups are drawn.
    group_masks : list of pd.Series
        A list of boolean masks identifying members of each group.
    confounder_cols : list of str
        The labels of the dataset columns corresponding to confounders.
    max_caliper : float
        The maximum allowed difference between the propensity scores of the observations.
        
    Returns
    -------
    matches : list of tuples of ints
        A list of ordered pairs of index values for respectively the treatment and control groups.
    """
    
    group_data = compute_propensity_scores(dataset, group_masks, confounder_cols)
    
    matches = []
    
    groups = [group_data[group_data['treatment'] == i] for i in group_data['treatment'].unique()]
    
    indices = np.argsort([len(group) for group in groups])
    groups = sorted(groups, key=len)
    smallest_group, search_groups = groups[0], groups[1:]
    smallest_indices, search_indices = list(smallest_group.index), list(map(lambda group: list(group.index), search_groups))
    
    def caliper_valid(propensities):
        pair_propensity_differences = np.vstack(list(map(lambda pair: abs(pair[0] - pair[1]), itertools.combinations(propensities[:], 2))))
        return (pair_propensity_differences < max_caliper).all()
    
    test_p = np.array([[1, 2, 3], [6, 5, 4], [7, 8, 9]])
    caliper_valid(test_p)
    
    for smallest_index in tqdm(smallest_indices):
        possible_match_index_tuples = itertools.product([smallest_index], *search_indices)
        match = next((possible_match for possible_match in possible_match_index_tuples if caliper_valid(np.vstack(list(map(lambda match, group: group.loc[match, 'propensity_scores'], possible_match, groups))))), None)
        if match:
            matches.append(match)
            for group_idx, search_index in enumerate(match[1:]):
                search_indices[group_idx].remove(search_index)
            
    print("Matched {} treatment observations out of max {} (success rate {:.2%}).".format(len(matches), len(smallest_group), len(matches) / len(smallest_group)))
    
    reordered_matches = []
    for match in matches:
        reordered_matches.append([match[i] for i in indices])
        
    return reordered_matches

def plot_distribution(df, target, hue, title=""):
    """ 
    Plots distributions (using a histogram and a KDE plot) for the reputation and ratings variable, using arxiv as a hue
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), dpi=80)
    # Reputation (don't plot KDE in same subplot as hist as it's harder to discern)
    sns.histplot(df, ax=ax[0], x=target, hue=hue, multiple="layer", stat="density", common_norm=False)
    sns.kdeplot(data=df, ax=ax[1], x=target, hue=hue, fill=True, common_norm=False)

    sns.despine(fig)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    plt.show()

def check_balance(dataset, matches, confounder_cols):
    """Visualize the confounder distributions across groups to check balance.
    
    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset from which the two groups are drawn.
    matches : list of (int, int)
        A list of ordered pairs of index values for respectively the treatment and control groups.
    confounder_cols : list of str
        The labels of the dataset columns corresponding to confounders.
    """
    groups = []
    for group_idx in range(len(matches[0])):
        group_indices = [match[group_idx] for match in matches]
        group = dataset.loc[group_indices]
        group['group'] = group_idx
        groups.append(group)
        
    matched = pd.concat(groups)
    
    for confounder_col in confounder_cols:
        title = "Balance test for confounder {}".format(confounder_col)
        
        plot_distribution(matched, target=confounder_col, hue="group", title=title)