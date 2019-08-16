#include <stdio.h>             
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_rng.h> 
#include <gsl/gsl_randist.h> 
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_blas.h>

#include "helper.h"
#include "particle.h"

#define VSET gsl_vector_set
#define VGET gsl_vector_get
#define MSET gsl_matrix_set
#define MGET gsl_matrix_get

//Read in data, initial value, innovation variance

void init(int d,gsl_vector *pxcurr,gsl_matrix *yobs,gsl_matrix *tunemat,gsl_vector *centerVec)
{
  int i;
  char *File1;
  FILE *f; 
  //Read in the data
  //File1="lv51errorsd1.dat"; //data set 1
  File1="lv26errorsd1.dat"; //data set 2
  f=fopen(File1,"r");
  gsl_matrix_fscanf(f,yobs);
  fclose(f); 
  //Read in the innovation variance
  //File1="tunematsd1.dat"; //use with data set 1
  File1="tunemat2sd1.dat";  //use with data set 2
  f=fopen(File1,"r");
  gsl_matrix_fscanf(f,tunemat);
  fclose(f);  
  //Read in the center
  //File1="centersd1b.dat"; //use with data set 1
  File1="center2sd1b.dat";  //use with data set 2
  f=fopen(File1,"r");
  gsl_vector_fscanf(f,centerVec);
  fclose(f);   
  //initialise params at center
  for(i=0;i<d;i++){
    VSET(pxcurr,i,VGET(centerVec,i));
  }
  //Innovation variance scaling
  gsl_matrix_scale(tunemat,1.31); //sd=1
}

//Evaluates difference in log prior

double prior(int d, gsl_vector *pxcurr, gsl_vector *pxstar)
{
  int j;
  double canprior=0.0, curprior=0.0;

  for (j=0;j<d;j++)
    {
      canprior+=log(gsl_ran_flat_pdf(VGET(pxstar,j),-8.0,8.0));
      curprior+=log(gsl_ran_flat_pdf(VGET(pxcurr,j),-8.0,8.0));
    }
  return(canprior-curprior); 
}

int main(int argc, char *argv[]) {
  int d=5; int count=0;
  //int no_obs=51,inter=1,dim=2; //data set 1
  int no_obs=26,inter=2,dim=2; //data set 2
  //int N=55; //AuxPF, data set 1
  //int N=150; //EnKF, data set 1 
  //int N=350; //AuxPF, data set 2
  int N=65; //EnKF, data set 2 
  long iters;
  double priorCont,mllprop,mllcurr,aprob,u;
  gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
  int i,j;
  gsl_matrix *yobs; gsl_matrix *tunemat; gsl_vector *centerVec;
  gsl_vector *pxstar; gsl_vector *pxcurr;
  iters=(long) atoi(argv[1]);  //iterations from command line
  
  tunemat=gsl_matrix_alloc(d,d); yobs=gsl_matrix_alloc(no_obs,dim); centerVec=gsl_vector_alloc(d);   
  pxstar=gsl_vector_alloc(d); pxcurr=gsl_vector_alloc(d); 
  gsl_rng_set(r,1234);
 
  //initialise
  mllcurr=-gsl_pow_int(10.0,5); //accept first
  init(d,pxcurr,yobs,tunemat,centerVec);
  for(i=0;i<d;i++){
    VSET(pxstar,i,VGET(pxcurr,i));
  }
 
  for(i=0;i<iters;i++){
    
    //random walk on log-scale
    mvn_sample(pxstar,pxcurr,tunemat,d,r);
    
    //evaluate prior contribution
    priorCont=prior(d,pxcurr,pxstar);
    //RUN PF
    //mllprop=particleAux(yobs,pxstar,dim,d,no_obs,N,inter,r); //auxiliary PF
    mllprop=particleEnKf(yobs,pxstar,dim,d,no_obs,N,inter,r); //Ensemble Kalman filter
    aprob=mllprop+priorCont-mllcurr; 
    u=gsl_ran_flat(r,0.0,1.0);
    
    if (log(u) < aprob){ 
      for(j=0;j<d;j++){
	VSET(pxcurr,j,VGET(pxstar,j));
      }
      mllcurr=mllprop;
      count+=1; //track chain move
    }
    //output
    printf("%7.7f %7.7f %7.7f %7.7f %7.7f %7.7f %7.7f %u \n",exp(VGET(pxcurr,0)),exp(VGET(pxcurr,1)),exp(VGET(pxcurr,2)),exp(VGET(pxcurr,3)),exp(VGET(pxcurr,4)),mllcurr,mllprop,count);
  }
  
  gsl_matrix_free(yobs);  
  gsl_matrix_free(tunemat);
  gsl_vector_free(centerVec);
  gsl_vector_free(pxstar);
  gsl_vector_free(pxcurr);

  return 0;
}

