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

#define VSET gsl_vector_set
#define VGET gsl_vector_get
#define MSET gsl_matrix_set
#define MGET gsl_matrix_get

//Auxiliay PF with bridge from Golightly & Wilkinson (2015)

double particleAux(gsl_matrix *yobs,gsl_vector *pxstar,int dim,int d,int no_obs,int N,gsl_rng *r)
{
  double margllprop=0.0;
  double wsum,pred,prey,llpprop,time,wsumcurr,wsumtarg,delt,delta,sig1,sig2,con;
  double rate,ratestar,u,c1,c2,c3,c4,c5,c6,c7,c8,mean,varInv;
  double tau,trackqprop,trackqcurr,adjust,dna,rna,prot,prot2;
  double mean1, mean2, var1, var2, mean1b, mean2b, var1b, var2b, covp;
  int i,j,k,l,p,rej;
  gsl_matrix *xprior;
  gsl_matrix *xsamples;
  gsl_vector *haz;
  gsl_vector *hazstar;
  gsl_vector *hazstarsum;
  gsl_vector *vec;
  gsl_vector *wts;
  gsl_ran_discrete_t *g;

  xprior=gsl_matrix_alloc(N,dim); 
  xsamples=gsl_matrix_alloc(N,dim);
  haz=gsl_vector_alloc(d);
  hazstar=gsl_vector_alloc(d);
  hazstarsum=gsl_vector_alloc(d);
  vec=gsl_vector_alloc(d);
  wts=gsl_vector_alloc(N);

  sig1=1.0;
  sig2=1.0; //data set D1
  //sig1=0.0;
  //sig2=0.0; //data set D2
  con=10.0;
  c1=exp(VGET(pxstar,0));  c2=exp(VGET(pxstar,1)); c3=exp(VGET(pxstar,2)); c4=exp(VGET(pxstar,3)); 
  c5=0.1; c6=0.9; c7=exp(VGET(pxstar,4));  c8=exp(VGET(pxstar,5));    

  gsl_vector_set_all(vec,0.0);

  for (i=0;i<no_obs-1;i++)
    {
      gsl_vector_set_all(wts,0.0);
      wsum=0.0;

      for(k=0;k<N;k++){
	trackqprop=0.0;
	trackqcurr=0.0;
	if(i==0){
	  dna = 5.0;
	  rna = 8.0;
	  prot = 8.0;
	  prot2 = 8.0;
	}
	else{
	  dna=MGET(xprior,k,0);
	  rna=MGET(xprior,k,1); 
	  prot=MGET(xprior,k,2);
	  prot2=MGET(xprior,k,3);  
	}
	llpprop=0.0;
	time=0.0;
	while(time < 1.0)
	  {
	    rate=0.0; ratestar=0.0;
	    VSET(haz,0,c1*dna*prot2); VSET(haz,1,c2*(con-dna)); VSET(haz,2,c3*dna); VSET(haz,3,c4*rna);
	    VSET(haz,4,c5*prot*(prot-1.0)*0.5); VSET(haz,5,c6*prot2); VSET(haz,6,c7*rna); VSET(haz,7,c8*prot);
	    for(l=0;l<d;l++){
	      rate+=VGET(haz,l);
	    }
	    delt=1.0-time;
	   
	    var1= delt*(c3*dna+c7*rna)+sig1*sig1;
	    var2= delt*(c4*rna+c8*prot+4.0*(c1*dna*prot2+c2*(con-dna)))+sig2*sig2;
	    mean1=(MGET(yobs,i+1,0)-rna-delt*(c3*dna-c7*rna))/var1; //dna mean
	    mean2=(MGET(yobs,i+1,1)-(prot+2.0*prot2)-delt*(c4*rna-c8*prot+2.0*c2*(con-dna)-2.0*c1*dna*prot2))/var2; //mean of P+2P_2
	    
	    VSET(vec,0,-2.0*VGET(haz,0)*mean2); VSET(vec,1,2.0*VGET(haz,1)*mean2);
	    VSET(vec,2,VGET(haz,2)*mean1); VSET(vec,3,VGET(haz,3)*mean2);
	    VSET(vec,6,-VGET(haz,6)*mean1); VSET(vec,7,-VGET(haz,7)*mean2);
	    for(l=0;l<d;l++){
	      VSET(hazstar,l,VGET(haz,l)+VGET(vec,l));
	      if(VGET(hazstar,l)<0.0000000000){
		VSET(hazstar,l,0.0); //truncate
	      }
	     
	      ratestar+=VGET(hazstar,l); //sum cond. haz.
	      VSET(hazstarsum,l,ratestar); //store partial sums
	    }
	    if(ratestar<=0.0){
	      //no more reactions
	      ratestar=0.0;
	      break;
	    }

	    u = gsl_ran_flat(r, 0.0, 1.0);
	    tau= -log(u)/ratestar;
	    time+=tau;
	    if(time > 1.0){
	      break;
	    }
	    u = gsl_ran_flat(r, 0.0, 1.0);
	    trackqcurr-= rate*tau;
	    trackqprop-= ratestar*tau; 
	    
	    if (u<(VGET(hazstarsum,0))/ratestar) 
	      {
		dna   -= 1;
		prot2 -= 1;
		trackqcurr+=log(VGET(haz,0)); trackqprop+=log(VGET(hazstar,0));
	      }else if (u<(VGET(hazstarsum,1))/ratestar) 
	      {
		dna   += 1;
		prot2 += 1;
		trackqcurr+=log(VGET(haz,1)); trackqprop+=log(VGET(hazstar,1));
	    }else if (u<(VGET(hazstarsum,2))/ratestar) 
	      {
		rna +=1;
		trackqcurr+=log(VGET(haz,2)); trackqprop+=log(VGET(hazstar,2));
	    }else if (u<(VGET(hazstarsum,3))/ratestar) 
	      {
		prot +=1;
		trackqcurr+=log(VGET(haz,3)); trackqprop+=log(VGET(hazstar,3));
	    }else if (u<(VGET(hazstarsum,4))/ratestar) 
	      {
		prot  -=2;
		prot2 +=1;
		trackqcurr+=log(VGET(haz,4)); trackqprop+=log(VGET(hazstar,4));
	    }else if (u<(VGET(hazstarsum,5))/ratestar) 
	      {
		prot  +=2;
		prot2 -=1;
		trackqcurr+=log(VGET(haz,5)); trackqprop+=log(VGET(hazstar,5));
	    }else if (u<(VGET(hazstarsum,6))/ratestar) 
	      {
		rna -=1;
		trackqcurr+=log(VGET(haz,6)); trackqprop+=log(VGET(hazstar,6));
	    }else
	      {
		prot-=1;
		trackqcurr+=log(VGET(haz,7)); trackqprop+=log(VGET(hazstar,7));
	    }
	    //printf("%u %u %7.7f %7.7f %7.7f %7.7f %7.7f \n",i,k,time,dna,rna,prot,prot2);
	    if(dna<0.0){
	      dna=0.0;
	      rej=1;
	      break;
	    }
	    if(rna<0.0){
	      rna=0.0;
	      rej=1;
	      break;
	    }
	  if(prot<0.0){
	    prot=0.0;
	    rej=1;
	    break;
	  }
	  if(prot2<0.0){
	    prot2=0.0;
	    rej=1;
	    break;
	  }
	  }
	
	if(ratestar<=0.0){
	  trackqcurr-= rate*(1.0-time);
	  trackqprop-= ratestar*(1.0-time); 
	}else{
	  trackqcurr-= rate*(1.0-(time-tau));
	  trackqprop-= ratestar*(1.0-(time-tau));  
	}
	
	llpprop=lnd(MGET(yobs,i+1,0),rna,sig1)+lnd(MGET(yobs,i+1,1),prot+2.0*prot2,sig2); 
	if(i==0){
	  llpprop=llpprop+lnd(MGET(yobs,0,0),8.0,sig1)+lnd(MGET(yobs,0,1),24.0,sig2);
	}
	// llpprop=0.0; //for sigma=0 case -- comment out the above 4 lines
	if(rej==0)
	  {
	    VSET(wts,k,exp((llpprop+trackqcurr-trackqprop)));
	  }else{
	  VSET(wts,k,0.0);
	}
	
	//sum weights
	wsum=wsum+VGET(wts,k);
	
	MSET(xsamples,k,0,dna);
	MSET(xsamples,k,1,rna);
	MSET(xsamples,k,2,prot);
	MSET(xsamples,k,3,prot2);
      }

      margllprop=margllprop+log(wsum);
        
      //systematic resampling
      k=0;
      u=gsl_ran_flat(r,0,1)/(double)N;  
      wsumcurr=VGET(wts,k)/wsum;
      wsumtarg=u;
      delta = 1.0/(double)N; 
      for(l=0;l<N;l++){
	while (wsumcurr<wsumtarg) {
	  k++;
	  wsumcurr += VGET(wts,k)/wsum;
	} 
	for(j=0;j<dim;j++){
	  MSET(xprior,l,j,MGET(xsamples,k,j));
	}	
	wsumtarg += delta;
      }
    }
  
  margllprop=margllprop-(no_obs-1.0)*log(N);
  gsl_matrix_free(xprior);
  gsl_matrix_free(xsamples);
  gsl_vector_free(haz);
  gsl_vector_free(hazstar);
  gsl_vector_free(hazstarsum); 
  gsl_vector_free(vec); 
  gsl_vector_free(wts);

  return margllprop;
}

//Ensemble Kalman filter

double particleEnKf(gsl_matrix *yobs,gsl_vector *pxstar,int dim,int d,int no_obs,int N,gsl_rng *r)
{
  double margllprop=0.0;
  double llpprop,time,sig1,sig2,con,cov;
  double rate,ratestar,u,c1,c2,c3,c4,c5,c6,c7,c8,mean,varInv;
  double tau,trackqprop,trackqcurr,adjust,dna,rna,prot,prot2;
  double mean1, mean2, var1, var2, mean1b, mean2b, var1b, var2b, covp;
  int i,j,k,l,p,rej;
  gsl_matrix *xprior;
  gsl_matrix *xsamples;
  gsl_vector *haz;
  gsl_vector *hazstar;
  gsl_vector *hazsum;
  gsl_matrix *disp_mat;
  gsl_matrix *disp; gsl_matrix *dispInv; gsl_matrix *mat; gsl_matrix *mat2;
  gsl_vector *vec; 
  gsl_vector *prom; gsl_vector *prom2; gsl_vector *pro; gsl_vector *obs; 
  gsl_vector_view a; gsl_vector_view b; 

  xprior=gsl_matrix_alloc(N,dim); 
  xsamples=gsl_matrix_alloc(N,dim);
  haz=gsl_vector_alloc(d);
  hazstar=gsl_vector_alloc(d);
  hazsum=gsl_vector_alloc(d);
  disp_mat=gsl_matrix_alloc(dim,dim);
  disp=gsl_matrix_alloc(2,2); dispInv=gsl_matrix_alloc(2,2); mat=gsl_matrix_alloc(4,2); mat2=gsl_matrix_alloc(4,2);
  vec=gsl_vector_alloc(dim); obs=gsl_vector_alloc(2);
  prom=gsl_vector_alloc(dim); prom2=gsl_vector_alloc(2); pro=gsl_vector_alloc(2);
  

  sig1=1.0;
  sig2=1.0; //data set D1
  //sig1=0.1;
  //sig2=0.1; //data set D2
  con=10.0;
  c1=exp(VGET(pxstar,0));  c2=exp(VGET(pxstar,1)); c3=exp(VGET(pxstar,2)); c4=exp(VGET(pxstar,3)); 
  c5=0.1; c6=0.9; c7=exp(VGET(pxstar,4));  c8=exp(VGET(pxstar,5));    

  gsl_vector_set_all(vec,0.0);

  for (i=0;i<no_obs-1;i++)
    {
      
      for(k=0;k<N;k++){
	trackqprop=0.0;
	
	if(i==0){
	  dna = 5.0;
	  rna = 8.0;
	  prot = 8.0;
	  prot2 = 8.0;
	}
	else{
	  dna=MGET(xprior,k,0);
	  rna=MGET(xprior,k,1); 
	  prot=MGET(xprior,k,2);
	  prot2=MGET(xprior,k,3);  
	}
	llpprop=0.0;
	time=0.0;
	while(time < 1.0)
	  {
	    rate=0.0; ratestar=0.0;
	    VSET(haz,0,c1*dna*prot2); VSET(haz,1,c2*(con-dna)); VSET(haz,2,c3*dna); VSET(haz,3,c4*rna);
	    VSET(haz,4,c5*prot*(prot-1.0)*0.5); VSET(haz,5,c6*prot2); VSET(haz,6,c7*rna); VSET(haz,7,c8*prot);
	    for(l=0;l<d;l++){
	      rate+=VGET(haz,l);
	      VSET(hazsum,l,rate); //store partial sums
	    }
	    if(rate<=0.0){
	      //no more reactions
	      rate=0.0;
	      break;
	    }

	    u = gsl_ran_flat(r, 0.0, 1.0);
	    tau= -log(u)/rate;
	    time+=tau;
	    if(time > 1.0){
	      break;
	    }
	    u = gsl_ran_flat(r, 0.0, 1.0);
	  
	    if (u<(VGET(hazsum,0))/rate) 
	      {
		dna   -= 1;
		prot2 -= 1;
		if(dna<0.0){
		  dna=0.0;
		}if(prot2<0.0){
		  prot2=0.0;
		}
		
	      }else if (u<(VGET(hazsum,1))/rate) 
	      {
		dna   += 1;
		prot2 += 1;
		
	    }else if (u<(VGET(hazsum,2))/rate) 
	      {
		rna +=1;
		
	    }else if (u<(VGET(hazsum,3))/rate) 
	      {
		prot +=1;
		
	    }else if (u<(VGET(hazsum,4))/rate) 
	      {
		prot  -=2;
		prot2 +=1;
		if(prot<0.0){
		  prot=0.0;
		}
		
	    }else if (u<(VGET(hazsum,5))/rate) 
	      {
		prot  +=2;
		prot2 -=1;
		if(prot2<0.0){
		  prot2=0.0;
		}
		
	    }else if (u<(VGET(hazsum,6))/rate) 
	      {
		rna -=1;
		if(rna<0.0){
		  rna=0.0;
		}
	      }else
	      {
		prot-=1;
		if(prot<0.0){
		  prot=0.0;
		}
	
	    }
	    
	  }
	MSET(xsamples,k,0,dna);
	MSET(xsamples,k,1,rna);
	MSET(xsamples,k,2,prot);
	MSET(xsamples,k,3,prot2);
      }
	
      //calculate mean and variance
      for (k = 0; k < dim; k++) {
	a = gsl_matrix_column (xsamples, k);
	VSET(prom,k,gsl_stats_mean(a.vector.data,a.vector.stride,a.vector.size));
	for (j = 0; j < (k+1); j++) {
	  b = gsl_matrix_column (xsamples, j);
	  cov = gsl_stats_covariance(a.vector.data, a.vector.stride,b.vector.data, b.vector.stride, a.vector.size);
	  MSET(disp_mat, k, j, cov); MSET(disp_mat, j, k, cov); 
	  
	}
      } 

      //update marginal likelihood
      MSET(disp,0,0,MGET(disp_mat,1,1)+sig1*sig1); MSET(disp,0,1,MGET(disp_mat,1,2)+2.0*MGET(disp_mat,1,3));
      MSET(disp,1,1,MGET(disp_mat,2,2)+4.0*MGET(disp_mat,2,3)+4.0*MGET(disp_mat,3,3)+sig2*sig2); MSET(disp,1,0,MGET(disp,0,1));
      VSET(prom2,0,VGET(prom,1)); VSET(prom2,1,VGET(prom,2)+2.0*VGET(prom,3));
      VSET(obs,0,MGET(yobs,i+1,0)); VSET(obs,1,MGET(yobs,i+1,1));

      trackqprop=lmgpdf(2,obs,prom2,disp);
      if(i==0){
	trackqprop=trackqprop+lnd(MGET(yobs,0,0),8.0,sig1)+lnd(MGET(yobs,0,1),24.0,sig2);
      }
     
      //calculate Kalman gain (disp2)
      invert2by2(disp,dispInv); //dispInv=(Sigma + R)^-1
      MSET(mat,0,0,MGET(disp_mat,0,1)); MSET(mat,0,1,MGET(disp_mat,0,2)+2.0*MGET(disp_mat,0,3));
      MSET(mat,1,0,MGET(disp_mat,1,1)); MSET(mat,1,1,MGET(disp_mat,1,2)+2.0*MGET(disp_mat,1,3));
      MSET(mat,2,0,MGET(disp_mat,2,1)); MSET(mat,2,1,MGET(disp_mat,2,2)+2.0*MGET(disp_mat,2,3));
      MSET(mat,3,0,MGET(disp_mat,3,1)); MSET(mat,3,1,MGET(disp_mat,3,2)+2.0*MGET(disp_mat,3,3)); // Sigma H
      matmul(dim,2,2,mat,dispInv,mat2);  //mat2 = 4by2 Kalman gain 
      //shift
      for(k=0;k<N;k++){
	VSET(pro,0,MGET(yobs,i+1,0)-MGET(xsamples,k,1)-sig1*gsl_ran_gaussian(r,1.0));
	VSET(pro,1,MGET(yobs,i+1,1)-MGET(xsamples,k,2)-2.0*MGET(xsamples,k,3)-sig2*gsl_ran_gaussian(r,1.0));
	matvecmul(dim,2,mat2,pro,vec);
	MSET(xprior,k,0,MGET(xsamples,k,0)+VGET(vec,0));
	MSET(xprior,k,1,MGET(xsamples,k,1)+VGET(vec,1));
	MSET(xprior,k,2,MGET(xsamples,k,2)+VGET(vec,2));
	MSET(xprior,k,3,MGET(xsamples,k,3)+VGET(vec,3));
	for(j=0;j<dim;j++){
	  if(MGET(xprior,k,j)<0.0){
	    MSET(xprior,k,j,0.000001);
	  }
	}
      }
      margllprop+=trackqprop;
    
    }
  
  
  gsl_matrix_free(xprior);
  gsl_matrix_free(xsamples);
  gsl_vector_free(haz);
  gsl_vector_free(hazstar);
  gsl_vector_free(hazsum); 
  gsl_matrix_free(disp_mat); gsl_matrix_free(mat); gsl_matrix_free(mat2);
  gsl_matrix_free(disp); gsl_matrix_free(dispInv);
  gsl_vector_free(vec); gsl_vector_free(obs); 
  gsl_vector_free(prom); gsl_vector_free(prom2); gsl_vector_free(pro);

  return margllprop;
}
