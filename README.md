*Monte_Carlo_Simulation.py contains the naive implementation of SRA using monte carlo simulation.
nMC=10**6 samples were randomly generated to simulate the parameters of the performance function
The simulation estimates a probability of failure of  0.002222 after calling the performance function 10**6 times.
*AK_MCS_U.py implements the first surrogate to the monte carlo simulation using active learning method of kriging. 
This implementation goes through the 10 stages discussed in the paper of Echard and al.
1/ The population S (set to 10**6) is generated following a normal distribution 
2/ N1 samples are randomly chosen, N1 is set to 12, to create the Domaine of Experience DoE. The performance function is evaluated in N1 points.
3/ The kriging model is fit to the N1 data points
4/ The kriging model is used to predict G_hat for the whole population S, which are the predcitions of the performance function by the kriging model.
5/ The active learning step consists of choosing the next best point to evaluate, with respect to the learning function U.
6/ Stopping criterion: Stop when the best learning value reach a maximum value of 2, i.e. min(U)>=2.
    7/else: evaluate the performance function on the next best point x* and add it to the DoE. go back to 3
    8/compute the coefficient of variance cov_pf. If it reaches the mthreshhold of 0.05, the AK_MCS has then converged. Go to 10.    
        9/ Add new data points to  the population using monte carlo samples, and go back to 4.
10/ print the results and end the iteration.
