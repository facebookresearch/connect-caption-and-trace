import torch
import scipy.optimize
import numpy as np
m = 10
pred = torch.rand([10, m, 4])
label = torch.rand([10, m, 4])

def local_OT(D):
    p = D.shape[1]; m = D.shape[2]
    # construct the cx, ax=b
    x = torch.rand([10,m*m])
    A = torch.zeros([m+m,m*m])
    b = torch.ones([m+m])
    for i in range(p):
        A[i, (i)*m:(i+1)*m] = 1
    for i in range(m):
        for j in range(p):
            A[p+i, j*m+i] = 1

    A_local = torch.zeros([m, m, m])
    for i in range(m):
        if i+2<=m-1:
            A_local[i, i, i+2:] = 1
        if i-2 >=0:
            A_local[i, i, :i-1] = 1
    A_local = A_local.view([m, m*m])
    b_local = torch.zeros([m])

    A = torch.cat([A, A_local], 0).numpy()
    b = torch.cat([b, b_local], 0).numpy()

    T_list = []
    for i in range(D.shape[0]):
        c = D[i].view(-1).detach().cpu().numpy()
        sol = scipy.optimize.linprog(c, A_eq = A, b_eq = b, bounds=(0, None))
        sol_x = torch.from_numpy(sol.x).view([p,m])
        T_list.append(sol_x)
    T = torch.stack(T_list, 0)
    return (T>0.5).float() # binarize it

T = local_OT(torch.rand([10,20,20]))

print('finish')