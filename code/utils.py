import numpy as np

def gen_data(B, V, N, alpha=0.1):
    assert B.shape[0] == B.shape[1]
    K = B.shape[0]
    alpha_vec = [alpha for _ in xrange(K)]
    
    pi = np.zeros((V, K))
    X = np.zeros((N, 3), dtype=int)
    for p in xrange(V):
        pi[p] = np.random.dirichlet(alpha_vec)
    
    for n in xrange(N):
        p = q = np.random.randint(V)
        while p == q:
            q = np.random.randint(V)
        
        p, q = sorted([p, q])
        g = np.random.choice(range(K), p=pi[p])
        h = np.random.choice(range(K), p=pi[q])
        y = np.random.binomial(1, p=B[g,h])
        X[n,:] = np.array([p,q,y])
        
    return X, pi


def gen_data_pr(V, B, N):
    X = np.zeros((N, 3), dtype=int)
    for n in xrange(N):
        p = q = np.random.randint(V)
        while p == q:
            q = np.random.randint(V)
        p, q = sorted([p, q])
        y = np.random.binomial(1, B)
        X[n,:] = np.array([p,q,y])
    return X


def make_B(K, probs=[.2]):
    B = np.eye(K) / 2. # 0.5 on diagonal
    for g in xrange(B.shape[0]):
        for h in xrange(g): # h < g, lower triangle
            B[h,g] = np.random.choice(probs)
            B[g,h] = 1 - B[h,g]
    return B
            
def get_interactions(X):
    V = max(X[:,1]) + 1
    I = np.zeros((V, V), dtype=int)
    for p, q, v in X:
        if v:
            I[p,q] += 1
        else:
            I[q,p] += 1
    return I