import numpy as np
import numpy as np
from scipy.stats import norm
import numpy as np
import math

def mcmc_algo(A, b, dim, lenchn, surr):
    U_start = A[1:2+dim-1]
    g_start = A[0]
    c = 0
    U_seed = U_start
    U = np.zeros((lenchn, dim))
    g = np.zeros(lenchn)

    while c < lenchn:
        c += 1
        t = np.zeros(dim)
        ratio = np.zeros(dim)
        alpha = np.zeros(dim)
        U_cand = np.zeros(dim)
        for k in range(dim):
            t[k] = (U_seed[k] - 1) + 2 * np.random.rand()
            ratio[k] = norm.pdf(t[k], 0, 1) / norm.pdf(U_seed[k], 0, 1)
            alpha[k] = min(1, ratio[k])
            
            if alpha[k] > np.random.rand():
                U_cand[k] = t[k]
            else:
                U_cand[k] = U_seed[k]

        gg = surr.predict(U_cand.reshape(1,-1))  
        
        if gg <= b:
            U[c-1, :] = U_cand
            g[c-1] = gg
        else:
            U[c-1, :] = U_seed
            if c == 1:
                g[c-1] = g_start
            else:
                g[c-1] = g[c-2]

        U_seed = U[c-1, :]

    return U, g







def SS(surr, dim, Nlevel):

 
    mu = np.array([0,0])  # mean of rvs
    sdev = np.array([1,1])  # sdev of rvs

    Nsim = 1
    p0 = 0.1

    nt = np.zeros(Nsim)
    pf = np.zeros(Nsim)
    cov_pf = np.zeros(Nsim)

    
    for k in range(Nsim):
        U = np.random.normal(0,1,size = (Nlevel,dim))

        g = np.zeros(Nlevel)
        g = surr.predict(U)
        A = np.column_stack((g, U))
        A = A[A[:,0].argsort()]
        b = []
        index = max(int(Nlevel * p0)-1, 0)
        b.append(A[index, 0])
      
        count = 0
        pf_cond = []
        pf_cond.append(p0)
        delta = []
        delta.append(np.sqrt((1-pf_cond[count])/(Nlevel*pf_cond[count])))

        sam = Nlevel
        lenchn = round(1/p0)
        gamma = [0]
        while b[count] > 0:
            for i in range(int(Nlevel*p0)):
                start_index = i * lenchn
                end_index = (i+1) * lenchn
                U[start_index:end_index, :], g[start_index:end_index] = mcmc_algo(A[i, :], b[count], dim, lenchn, surr) # assuming mcmc_algo is a function you have elsewhere
            A = np.column_stack((g, U))
            A = A[A[:,0].argsort()]
            count += 1
           

            b.append(A[max(math.floor(Nlevel * p0)-1, 0), 0])

            if b[count] <= 0:
                b[count] = 0
                II = g <= b[count]
                pf_cond.append(np.sum(II) / Nlevel)
            else:
                II = g <= b[count]
                pf_cond.append(p0)

            R = np.zeros(lenchn)

            for klg in range(lenchn):
                sumind = np.zeros(int(Nlevel*p0))
                for j in range(int(Nlevel*p0)):
                    failsamp = II[j*lenchn:(j+1)*lenchn]
                    for l in range(lenchn-klg):
                        sumind[j] += failsamp[l] * failsamp[l+klg]
                R[klg] = np.sum(sumind) / (Nlevel - klg * Nlevel * p0) - pf_cond[count] ** 2

            rho = R[1:] / R[0]
           
            gamma.append(2 * np.sum(rho * (1 - p0 * np.arange(1, lenchn))))

            if pf_cond[count] == 1:
                delta.append(0)
            else:
                delta.append(np.sqrt((1-pf_cond[count])/(Nlevel*pf_cond[count])*(1+gamma[count])))

            sam += Nlevel

        nt[k] = sam
        pf[k] = np.prod(pf_cond)
        cov_pf[k] = np.sqrt(np.sum(np.array(delta)**2))
        gamma_mean =  np.mean(gamma)

    mean_pf = np.mean(pf)
    cov_avg = np.std(pf)/mean_pf
    cov_estimated = np.mean(cov_pf)
    print("Mean PF:", mean_pf)
    print("COV Average:", cov_avg)
    print("COV Estimated:", cov_estimated)
    return mean_pf, cov_estimated, gamma_mean



