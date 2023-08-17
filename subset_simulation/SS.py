import numpy as np
import numpy as np
from scipy.stats import norm
import numpy as np

def gcal1(X,mu,sdev):
   
    return 0.5 * (X[0]-2)**2 - 1.5 * (X[1]-5)**3 -3


def gcal2(X,mu,sdev):
    k = 6
    term1 = 3 + 0.1 * (X[0] - X[1])**2 - (X[0] + X[1])/(np.sqrt(2))
    term2 = 3 + 0.1 * (X[0] - X[1])**2 + (X[0] + X[1])/(np.sqrt(2))
    term3 = (X[0] - X[1]) + k / (2**0.5)
    term4 = (X[1] - X[0]) + k / (2**0.5)
    
    return min(term1, term2, term3, term4)
def mcmc_algo(A, b, dim, lenchn, mu, sdev):
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

        gg = gcal2(U_cand, mu, sdev)  
        
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







def main():
    np.random.seed()  # Equivalent of MATLAB's rng('shuffle')

    dim = 2
    mu = np.array([0,0])  # mean of rvs
    sdev = np.array([1,1])  # sdev of rvs

    Nsim = 10
    Nlevel = 1000
    p0 = 0.1

    nt = np.zeros(Nsim)
    pf = np.zeros(Nsim)
    cov_pf = np.zeros(Nsim)

    
    for k in range(Nsim):
        U = np.random.normal(0,1,size = (Nlevel,dim))
        g = np.zeros(Nlevel)
        for i in range(Nlevel):
            g[i] = gcal2(U[i],mu,sdev)
        A = np.column_stack((g, U))
        A = A[A[:,0].argsort()]
        b = np.zeros(Nlevel)
        b[0] = A[int(Nlevel * p0), 0]
      
        count = 0
        pf_cond = np.ones(Nlevel)
        pf_cond[count] = p0
        delta = np.zeros(Nlevel)
        delta[count] = np.sqrt((1-pf_cond[count])/(Nlevel*pf_cond[count]))

        sam = Nlevel
        lenchn = round(1/p0)

        while b[count] > 0:
            for i in range(int(Nlevel*p0)):
                start_index = i * lenchn
                end_index = (i+1) * lenchn
                U[start_index:end_index, :], g[start_index:end_index] = mcmc_algo(A[i, :], b[count], dim, lenchn, mu, sdev) # assuming mcmc_algo is a function you have elsewhere
            
            A = np.column_stack((g, U))
            A = A[A[:,0].argsort()]
            count += 1
            b[count] = A[int(Nlevel*p0), 0]

            if b[count] <= 0:
                b[count] = 0
                II = g <= b[count]
                pf_cond[count] = np.sum(II) / Nlevel
            else:
                II = g <= b[count]
                pf_cond[count] = p0

            R = np.zeros(lenchn)

            for klg in range(lenchn):
                sumind = np.zeros(int(Nlevel*p0))
                for j in range(int(Nlevel*p0)):
                    failsamp = II[j*lenchn:(j+1)*lenchn]
                    for l in range(lenchn-klg):
                        sumind[j] += failsamp[l] * failsamp[l+klg]
                R[klg] = np.sum(sumind) / (Nlevel - klg * Nlevel * p0) - pf_cond[count] ** 2

            rho = R[1:] / R[0]
            gamma = np.zeros(Nlevel)
            gamma[count] = 2 * np.sum(rho * (1 - p0 * np.arange(1, lenchn)))

            if pf_cond[count] == 1:
                delta[count] = 0
            else:
                delta[count] = np.sqrt((1-pf_cond[count])/(Nlevel*pf_cond[count])*(1+gamma[count]))

            sam += Nlevel

        nt[k] = sam
        pf[k] = np.prod(pf_cond)
        cov_pf[k] = np.sqrt(np.sum(delta**2))

    mean_pf = np.mean(pf)
    cov_avg = np.std(pf)/mean_pf
    cov_estimated = np.mean(cov_pf)
    print("Mean PF:", mean_pf)
    print("COV Average:", cov_avg)
    print("COV Estimated:", cov_estimated)

# You can call main() or wrap this inside another function, depending on your needs
main()
