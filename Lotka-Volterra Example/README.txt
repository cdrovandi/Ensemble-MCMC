Software to accompany: 

Drovandi et al. (2019)

Lotka-Volterra example

helper.c	 helper functions 
particle.c       Auxiliary PF and ensemble Kalman filter
mainLV.c	 code to implement PMMH for LV 
lv51errorsd1.dat 51 observations on predator and prey levels, corrupted by addive Gaussian noise (with sd=1)
lv26errorsd1.dat 26 observations on predator and prey levels, corrupted by addive Gaussian noise (with sd=1)
tunematsd1.dat   Estimate of posterior variance from pilot run (data set 1)
centersd1b.dat   Estimated posterior median from pilot run (data set 1)
tunemat2sd1.dat  Estimate of posterior variance from pilot run (data set 2)
center2sd1.dat   Estimated posterior median from pilot run (data set 2)

Compile with

gcc mainLV.c particle.c helper.c -lgsl -lgslcblas -lm -o out
./out 1000


Email:

a.golightly@ncl.ac.uk

