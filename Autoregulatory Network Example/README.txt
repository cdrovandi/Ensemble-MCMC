Software to accompany: 

Drovandi et al (2019)

Autoregulatory network example

helper.c	         helper functions 
particle.c               Auxiliary PF and ensemble Kalman filter
mainAR.c	         code to implement PMMH for autoregulatory network 
rnaprotCOMB101oev1-1.dat 101 observations on rna and total protein levels, corrupted by addive Gauusian noise (with sd=1)
rnaprotCOMB101oev0.dat   101 observations on rna and total protein levels, exact observation regime
tunemat101.dat           Estimate of posterior variance from pilot run 
center101.dat            Estimated posterior median from pilot run 


Compile with

gcc mainAR.c particle.c helper.c -lgsl -lgslcblas -lm -o out
./out 1000


Email:

a.golightly@ncl.ac.uk

