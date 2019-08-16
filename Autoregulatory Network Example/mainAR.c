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
  File1="rnaprotCOMB101oev1-1.dat"; // D1 - 101 obs at int times on rna and total protein counts (sigma=1)
  //File1="rnaprotCOMB101oev0.dat"; // D2 - 101 obs at int times on rna and total protein counts (sigma=0)
  f=fopen(File1,"r");
  gsl_matrix_fscanf(f,yobs);
  fclose(f); 
  //Read in the innovation variance
  File1="tunemat101.dat";
  f=fopen(File1,"r");
  gsl_matrix_fscanf(f,tunemat);
  fclose(f);  
  //Read in the center
  File1="center101.dat";
  f=fopen(File1,"r");
  gsl_vector_fscanf(f,centerVec);
  fclose(f);   
  //initialise params at center
  for(i=0;i<d;i++){
    VSET(pxcurr,i,VGET(centerVec,i));
  }
  //Innovation variance scaling
  gsl_matrix_scale(tunemat,0.9);  
 
}

//Evaluates difference in log prior

double prior(int d, gsl_vector *pxcurr, gsl_vector *pxstar)
{
  int j;
  double canprior=0.0, curprior=0.0;

  for(j=0;j<d;j++)
    {
      canprior+=log(gsl_ran_gamma_pdf(exp(VGET(pxstar,j)),1.0,1.0/2.0))+VGET(pxstar,j);
      curprior+=log(gsl_ran_gamma_pdf(exp(VGET(pxcurr,j)),1.0,1.0/2.0))+VGET(pxcurr,j);
    }

  return(canprior-curprior); 
}

int main(int argc, char *argv[]) {
  int d=6,drate=8; int count=0;
  int no_obs=101,dim=4;
  //int N=200; //EnKf data set D1
  int N=400; //Aux PF data set D1
  //int N=2000; //EnKf data set D2
  //int N=370; //Aux PF data set D2 
  long iters;
  double priorCont,mllprop,mllcurr,aprob,u;
  double aprob1;
  gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
  int i,j,m;
  gsl_vector *pxstar; gsl_vector *pxcurr;
  gsl_matrix *yobs;
  gsl_matrix *tunemat;  
  gsl_vector *centerVec;
  iters=(long) atoi(argv[1]);
  tunemat=gsl_matrix_alloc(d,d);  
  yobs=gsl_matrix_alloc(no_obs,2);
  centerVec=gsl_vector_alloc(d);   
  pxstar=gsl_vector_alloc(d); pxcurr=gsl_vector_alloc(d); 
  gsl_rng_set(r,1234);
 
  //initialise
  mllcurr=-1000000.0;
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
    //mllprop=particleEnKf(yobs,pxstar,dim,drate,no_obs,N,r);  //EnKF 
    mllprop=particleAux(yobs,pxstar,dim,drate,no_obs,N,r); //Aux PF
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
    printf("%7.7f %7.7f %7.7f %7.7f %7.7f %7.7f %7.7f %7.7f %u \n",exp(VGET(pxcurr,0)),exp(VGET(pxcurr,1)),exp(VGET(pxcurr,2)),exp(VGET(pxcurr,3)),exp(VGET(pxcurr,4)),exp(VGET(pxcurr,5)),mllcurr,mllprop,count);
  }

  gsl_matrix_free(yobs);  
  gsl_matrix_free(tunemat);
  gsl_vector_free(centerVec);
  gsl_vector_free(pxstar);
  gsl_vector_free(pxcurr);
  
  return 0;
}
