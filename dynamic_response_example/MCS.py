import numpy as np
def g(c1, c2, m, r, t1, F1):
    w0 = np.sqrt((c1 * c2)/m)
    return 3 * r - np.abs(2 * F1 * np.sin(w0*t1/2)/ (m*w0**2))

nMC = 700000
pf_values = []
for i in range(30):
    m = np.random.normal(1, 0.05, size=nMC)
    c1 = np.random.normal(1, 0.1, size=nMC)
    c2 = np.random.normal(0.1, 0.01, size=nMC)
    r = np.random.normal(0.5, 0.05, size=nMC)
    F1 = np.random.normal(1, 0.2, size=nMC)
    t1 = np.random.normal(1, 0.2, size=nMC) 

    failure_count = 0 

    for i in range(nMC):
        if g(c1[i], c2[i], m[i], r[i], t1[i], F1[i]) >= 0:
                failure_count += 1

            # Calculate the probability of failure
    Pf = failure_count / nMC
    pf_values.append(Pf)

print(np.mean(pf_values))
print(np.std(pf_values))
