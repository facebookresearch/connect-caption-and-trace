import torch
import scipy.optimize
import numpy as np

def local_OT(D, window = 0):
    window = window
    p = D.shape[1]; m = D.shape[2]  # p < m, e.g., p = 10, m = 20
    # construct the cx, ax=b
    x = torch.rand([10,p*m])
    A = torch.zeros([p,p*m])
    b = torch.ones([p])
    for i in range(p):
        A[i, (i)*m:(i+1)*m] = 1

    G = torch.zeros([m, p*m])
    for i in range(m):
        for j in range(p):
            G[i, j*m+i] = 1
    h = torch.ones([m])

    A_local = torch.zeros([p, p, m])
    for i in range(p):
        # left = np.floor((i - window) * (m*1.0/p))
        # right = (i + window) * (m*1.0/p)
        left = (i - window) * (m * 1.0 / p)
        right = (i + 1 + window) * (m * 1.0 / p)
        for j in range(m):
            # if j < left or j >= right:
            if j < left or j >= right:
                A_local[i, i, j] = 1
        # if i+window+1<=m-1:
        #     A_local[i, i, i+(window+1):] = 1
        # if i-(window+1) >=0:
        #     A_local[i, i, :i-window] = 1
    A_local = A_local.view([p, p*m])
    b_local = torch.zeros([p])

    A = torch.cat([A, A_local], 0).numpy()
    b = torch.cat([b, b_local], 0).numpy()
    G = G.numpy()
    h = h.numpy()

    T_list = []
    for i in range(D.shape[0]):
        c = D[i].view(-1).detach().cpu().numpy()
        try:
            sol = scipy.optimize.linprog(c, A_ub = G, b_ub = h, A_eq = A, b_eq = b, bounds=(0, 1)) #options={'maxiter': 200, 'sym_pos':False}
            sol_x = torch.from_numpy(sol.x).view([p,m]).float()
        except:
            sol_x = torch.cat([torch.eye(p), torch.zeros(p, m-p)], 1)
        T_list.append(sol_x)
    T = torch.stack(T_list, 0)
    return T.to(D.device) #(D * T.cuda()).sum() / p #(T>0.5).float() # binarize it

### for debug
# D = torch.rand([1, 10, 20])
# cost_orig = torch.diag(D[0]).sum()
# T = local_OT(D)
# cost_new = (D * T).sum()
# print(cost_orig, cost_new)
