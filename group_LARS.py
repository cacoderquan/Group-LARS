#!/usr/bin/python
import numpy as np
from scipy import linalg

   

def group_LARS(X, y, groups, p):
    """
    Group least angle regression selection algorithm:
    
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Design Matrix.
    y : array of shape (n_samples,)        
    groups : array of shape (n_features,)
        Group label. For each column, it indicates
        its group apertenance.
    p : array of float numbers
        parameters defining the kernal matrices
    Returns
    -------
    beta : array
        vector of coefficients
    References
    ----------
    "Model selection and estimation in regression with grouped variables",
    Ming Yuan and Yi Lin
    """
    
    # .. local variables ..
    X, y, groups, p = map(np.asanyarray, (X, y, groups, p))
    if len(groups) != X.shape[1]:
        raise ValueError( "Incorrect shape for groups" )
    if np.count_nonzero(p) < len(p):
        raise ValueError( "Kernal matrices cannot be zero" )       
    
    # .. use integer indices for groups ..
    group_labels = [np.where(groups == i)[0] for i in np.unique(groups) ]    
    number_of_groups = len(group_labels)  # the number of groups
    
    if len(p) != number_of_groups:
        raise ValueError( "number of groups is different from number of kernal matrices" )
      
    # Step 1: start from beta = 0, k = 1 and r = y
    set_index = 0  # index of the group with the maximum correlation with residue
    max_corr = 0
    r = y
    k = 0
    beta = np.zeros(X.shape[1], dtype=X.dtype) 
    corr_index_set = []  #'most correlated set' belongs to {1,2,...,J}
    
    # Step 2: compute the current 'most correlated set' A_1
    for i in range(number_of_groups):
        # slicing 
        X_i = X[:, group_labels[i]]
        projection = linalg.norm(np.dot( X_i.T, r )) ** 2 / p[i]
        if projection > max_corr:
            max_corr = projection
            set_index = i
    corr_index_set.append(set_index)
    
    alpha = 0
    corr_set = []  # corr_set is the set A_k
    
    while alpha != 1 and k <= number_of_groups + 1:
        # terminate when all factors have been added
        if len(corr_index_set) == number_of_groups:
            alpha = 1
            continue
        
        # compute set A_k
        for i in corr_index_set:
            corr_set = corr_set + group_labels[i].tolist()
        corr_set = list(set(corr_set))
            
        # compute gamma_k
        X_k = X[:, corr_set]
        H_k = np.dot(X_k.T, X_k) 
        gamma_k = np.dot(linalg.inv(H_k), np.dot(X_k.T, r))
        
        # Step 3: compute the "direction vector" gamma
        j = 0
        gamma = np.zeros(len(groups))
        for i in corr_set:
            gamma[i] = gamma_k[j]
            j = j + 1
            
        # Step 4: calculate alpha_i
        alphaset = []
        alpha = 1 
        i_star = -1   # the factor will be added
        for i in range(number_of_groups):  # the number of groups
            if i not in corr_index_set:   #'most correlated set' belong to {1,2,...,J}
                j = corr_index_set[0]
                # solve the quadratic equation
                X_i = X[:, group_labels[i]]   #X_j
                X_j = X[:, group_labels[j]]   #X_{j'}
                
                A_i = np.dot(X_i.T, r)
                A_j = np.dot(X_j.T, r)
                B_i = np.dot(np.dot(X_i.T, X), gamma)
                B_j = np.dot(np.dot(X_j.T, X), gamma)
                
    
                a = linalg.norm(B_i) ** 2 / p[i] - linalg.norm(B_j) ** 2 / p[j]
                b = -2 * linalg.norm(np.multiply(A_i, B_i), 1) / p[i] + 2 * linalg.norm(np.multiply(A_j, B_j), 1) / p[j]
                c = linalg.norm(A_i) ** 2 / p[i] - linalg.norm(A_j) ** 2 / p[j]
                
                if b ** 2 < 4 * a * c:
                    print "This equation has no real solution"
                    continue
                
                coeff = [a, b, c]
                sol = np.roots(coeff)  # solutions of the quadratic equation a * x^2 + b * x + c = 0
                alpha_i = 1
                if len(sol) == 2:
                    min_elem = min(sol.tolist())
                    max_elem = max(sol.tolist())
                    if min_elem >= 0 and min_elem <= 1:
                        alpha_i = min_elem
                    elif max_elem >= 0 and max_elem <= 1:
                        alpha_i = max_elem
                elif len(sol) == 1:
                    alpha_i = sol[0]
                    
                # calculate the step size
                if alpha_i < alpha:
                    alpha = alpha_i
                    i_star = i
                    
        # Step 5: add the factor
        if i_star != -1:
            corr_index_set = corr_index_set + [i_star]
        
        # Step 6: update beta and residual
        beta = beta + alpha * gamma  
        r = y - np.dot(X, beta)
        k = k + 1
        
    return beta
                
            
if __name__ == '__main__':
    from sklearn import datasets
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    groups = np.r_[[0, 0], np.arange(X.shape[1] - 2)]
    group_labels = [np.where(groups == i)[0] for i in np.unique(groups) ]  
    p = np.ones(len(group_labels))   
   # p = np.arange(1.0,9.0)
    coefs = group_LARS(X, y, groups, p)
    print "coefs", coefs
   