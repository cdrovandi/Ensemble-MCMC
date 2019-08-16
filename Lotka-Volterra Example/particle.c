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

double particleAux(gsl_matrix *yobs,gsl_vector *pxstar,int dim,int d,int no_obs,int N,double inter,gsl_rng *r)
{
  double margllprop=0.0;
  double wsum,pred,prey,llpprop,time,wsumcurr,wsumtarg,delt,delta;
  double rate,ratestar,u,c1,c2,c3,c4,c5,trackqprop,trackqcurr,tau;
  int i,j,k,l,index,reject;
  gsl_vector *wts; 
  gsl_vector *pxstarE;
  gsl_matrix *xprior;
  gsl_matrix *xsamples;
  gsl_matrix *disp_mat;
  gsl_matrix *disp; 
  gsl_matrix *hst_mat;  
  gsl_vector *pro;
  gsl_vector *prom;
  gsl_vector *haz;
  gsl_vector *hazstar;
  gsl_vector *driftVec;
  gsl_vector *vec;
  gsl_vector *vec2;
  gsl_vector *vec3;
  gsl_ran_discrete_t *g;

  wts=gsl_vector_alloc(N);
  pxstarE=gsl_vector_alloc(d);
  xprior=gsl_matrix_alloc(N,dim); 
  xsamples=gsl_matrix_alloc(N,dim);
  disp_mat=gsl_matrix_alloc(dim,dim);
  disp=gsl_matrix_alloc(dim,dim);
  hst_mat=gsl_matrix_alloc(d,dim);
  pro=gsl_vector_alloc(dim);
  prom=gsl_vector_alloc(dim);
  driftVec=gsl_vector_alloc(dim);
  haz=gsl_vector_alloc(d);
  hazstar=gsl_vector_alloc(d);
  vec=gsl_vector_alloc(dim);
  vec2=gsl_vector_alloc(dim);
  vec3=gsl_vector_alloc(d);

  c1=exp(VGET(pxstar,0)); VSET(pxstarE,0,c1);
  c2=exp(VGET(pxstar,1)); VSET(pxstarE,1,c2);
  c3=exp(VGET(pxstar,2)); VSET(pxstarE,2,c3);
  c4=exp(VGET(pxstar,3)); 
  c5=exp(VGET(pxstar,4));

  for (i=0;i<no_obs-1;i++)
    {
      gsl_vector_set_all(wts,0.0);
      wsum=0.0;
      for(k=0;k<N;k++){
	trackqprop=0.0;
	trackqcurr=0.0;

	if(i==0){
	  prey=71.0;
	  pred=79.0;
	}
	else{
	  index=k;
	  prey=MGET(xprior,index,0);
	  pred=MGET(xprior,index,1); 
	}
	llpprop=0.0;
	time=0.0;
        reject=0;
	while(time < inter)
	  {
	    rate=0.0;
	    ratestar=0.0;
	    VSET(haz,0,c1*prey); VSET(haz,1,c2*prey*pred); VSET(haz,2,c3*pred);
	    for(l=0;l<d;l++){
	      rate+=VGET(haz,l);
	    }
	    delt=inter-time;
	    VSET(pro,0,prey); VSET(pro,1,pred); //xt
	    drift(driftVec,pro,pxstarE,delt); //Shdelt
	    diffusion(disp_mat,pro,pxstarE,delt); //SdiaghSTdelt
	    hst(hst_mat,pro,pxstarE); //diaghST
	    MSET(disp_mat,0,0,MGET(disp_mat,0,0)+c4*c4);
	    MSET(disp_mat,1,1,MGET(disp_mat,1,1)+c5*c5);
	    invert2by2(disp_mat,disp); //disp=(sig+SdiaghSTdelt)^-1
	    for(l=0;l<dim;l++){
	      VSET(vec,l,MGET(yobs,i+1,l)-VGET(pro,l)-VGET(driftVec,l)); //y-xt-Shdelt
	    }
	    
	    matvecmul(dim,dim,disp,vec,vec2); //vec2=(sig+SdiaghSTdelt)^-1 %*% (y-xt-Shdelt)
	    matvecmul(d,dim,hst_mat,vec2,vec3); //vec3=diaghST %*% vec2
	    for(l=0;l<d;l++){
	      VSET(hazstar,l,VGET(haz,l)+VGET(vec3,l));
	      if(VGET(hazstar,l)<0.0){
		VSET(hazstar,l,0.0); //truncate
	      }
	    }
	   
	    for(l=0;l<d;l++){
	      ratestar+=VGET(hazstar,l); //combined cond. haz.
	    }
	    if(ratestar<=0.0){
	      prey=0.0; pred=0.0; time=inter; rate=0.0; ratestar=0.0; //extinct
	      break;
	    }
	    
	    u = gsl_ran_flat(r, 0.0, 1.0);
	    tau=-log(u)/ratestar;
	    time+= tau;
	    if(time > inter){
	      break;
	    }
	    u = gsl_ran_flat(r, 0.0, 1.0);
	    trackqcurr-= rate*tau;
	    trackqprop-= ratestar*tau; 
	    if(u<(VGET(hazstar,0))/ratestar)
	      {
		prey   += 1;
		trackqcurr+=log(VGET(haz,0));
		trackqprop+=log(VGET(hazstar,0));
	      }else if(u<(VGET(hazstar,0)+VGET(hazstar,1))/ratestar)
	      {
		prey   -= 1;
		pred   += 1;
		trackqcurr+=log(VGET(haz,1));
		trackqprop+=log(VGET(hazstar,1));
	      }else
	      {
		pred -=1;
		trackqcurr+=log(VGET(haz,2));
		trackqprop+=log(VGET(hazstar,2));
	      }
	    if((prey<0.0)||(pred<0.0)){
	      reject=1;
	      break;
	    }
	  }
	
	trackqcurr-= rate*(inter-(time-tau));
	trackqprop-= ratestar*(inter-(time-tau));  
	
	llpprop=lnd(prey,MGET(yobs,i+1,0),c4)+lnd(pred,MGET(yobs,i+1,1),c5);
	if(i==0){
	  llpprop=llpprop+lnd(MGET(yobs,0,0),71.0,c4)+lnd(MGET(yobs,0,1),79.0,c5);
	}
	if(reject==0){
	  VSET(wts,k,exp((llpprop+trackqcurr-trackqprop)));
	}
	
	//sum weights
	wsum=wsum+VGET(wts,k);
	
	MSET(xsamples,k,0,prey);
	MSET(xsamples,k,1,pred);	
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
  gsl_matrix_free(disp_mat);
  gsl_matrix_free(disp);
  gsl_matrix_free(hst_mat);
  gsl_vector_free(pro);
  gsl_vector_free(prom);
  gsl_vector_free(haz);
  gsl_vector_free(hazstar);
  gsl_vector_free(vec);
  gsl_vector_free(vec2);
  gsl_vector_free(vec3);
  gsl_vector_free(driftVec);
  gsl_vector_free(wts);
  gsl_vector_free(pxstarE);
 
  return margllprop;
}


//Ensemble Kalman filter

double particleEnKf(gsl_matrix *yobs,gsl_vector *pxstar,int dim,int d,int no_obs,int N,double inter,gsl_rng *r)
{
  double margllprop=0.0;
  double pred,prey,time,delt,cov;
  double rate,u,c1,c2,c3,c4,c5,trackqprop,tau;
  int i,j,k,l,index,reject;
  gsl_vector_view a; gsl_vector_view b;
  gsl_matrix *xprior; gsl_matrix *xsamples;
  gsl_matrix *disp_mat;
  gsl_matrix *disp; gsl_matrix *dispInv; gsl_matrix *disp2; 
  gsl_vector *pro;
  gsl_vector *prom;
  gsl_vector *obs;
  gsl_vector *vec;
  gsl_vector *haz;

  xprior=gsl_matrix_alloc(N,dim); 
  xsamples=gsl_matrix_alloc(N,dim);
  disp_mat=gsl_matrix_alloc(dim,dim);
  disp=gsl_matrix_alloc(dim,dim); dispInv=gsl_matrix_alloc(dim,dim); disp2=gsl_matrix_alloc(dim,dim);
  pro=gsl_vector_alloc(dim); prom=gsl_vector_alloc(dim); obs=gsl_vector_alloc(dim);
  vec=gsl_vector_alloc(dim);
  haz=gsl_vector_alloc(d);
 
  c1=exp(VGET(pxstar,0)); 
  c2=exp(VGET(pxstar,1)); 
  c3=exp(VGET(pxstar,2)); 
  c4=exp(VGET(pxstar,3)); 
  c5=exp(VGET(pxstar,4));
  
  for (i=0;i<no_obs-1;i++)
    {
      for(k=0;k<N;k++){
	trackqprop=0.0;

	if(i==0){
	  prey=71.0; pred=79.0;
	}
	else{
	  index=k;
	  prey=MGET(xprior,index,0);
	  pred=MGET(xprior,index,1); 
	}
	time=0.0;
        reject=0;
	//Run model forward
	while(time < inter)
	  {
	    rate=0.0;
	    VSET(haz,0,c1*prey); VSET(haz,1,c2*prey*pred); VSET(haz,2,c3*pred);
	    for(l=0;l<d;l++){
	      rate+=VGET(haz,l);
	    }
	    
	    if(rate >0){
	      u = gsl_ran_flat(r, 0.0, 1.0);
	      time += -log(u)/rate;
	      if(time > inter)
		break;
	      u = gsl_ran_flat(r, 0.0, 1.0);
	      if(u<(VGET(haz,0))/rate)
		{
		  prey   += 1;
		}else if(u<(VGET(haz,0)+VGET(haz,1))/rate){
		prey   -= 1;
		pred   += 1;
	      }else{
		pred -=1;
	      }
	      if(prey<0.0){
		prey=0.0;
	      }
	      if(pred<0.0){
		pred=0.0;
	      }
	    }else{ //all dead
	      time=inter;
	    }
	  }
	
	MSET(xsamples,k,0,prey);
	MSET(xsamples,k,1,pred);
	
      }
      
      //calculate mean and variance
	for (k = 0; k < dim; k++) {
	  a = gsl_matrix_column (xsamples, k);
	  VSET(prom,k,gsl_stats_mean(a.vector.data,a.vector.stride,a.vector.size));
	  for (j = 0; j < (k+1); j++) {
	    b = gsl_matrix_column (xsamples, j);
	    cov = gsl_stats_covariance(a.vector.data, a.vector.stride,b.vector.data, b.vector.stride, a.vector.size);
	    MSET(disp_mat, k, j, cov); MSET(disp_mat, j, k, cov); 
	    //printf("%u %u %7.7f %7.7f \n",k,j,MGET(disp_mat,k,j),MGET(disp_mat,j,k));
	  }
	}

      //update marginal likelihood
      MSET(disp,0,0,MGET(disp_mat,0,0)+c4*c4); MSET(disp,0,1,MGET(disp_mat,0,1));
      MSET(disp,1,1,MGET(disp_mat,1,1)+c5*c5); MSET(disp,1,0,MGET(disp_mat,1,0));
      VSET(obs,0,MGET(yobs,i+1,0)); VSET(obs,1,MGET(yobs,i+1,1));
      trackqprop=lmgpdf(dim,obs,prom,disp);
      if(i==0){
	trackqprop=trackqprop+lnd(MGET(yobs,0,0),71.0,c4)+lnd(MGET(yobs,0,1),79.0,c5);
      }

      //calculate Kalman gain (disp2)
      invert2by2(disp,dispInv); //dispInv=(Sigma + R)^-1
      matmul(dim,dim,dim,disp_mat,dispInv,disp2); 

      //shift
      for(k=0;k<N;k++){
	VSET(pro,0,MGET(yobs,i+1,0)-MGET(xsamples,k,0)-c4*gsl_ran_gaussian(r,1.0));
	VSET(pro,1,MGET(yobs,i+1,1)-MGET(xsamples,k,1)-c5*gsl_ran_gaussian(r,1.0));
	matvecmul(dim,dim,disp2,pro,vec);
	MSET(xprior,k,0,MGET(xsamples,k,0)+VGET(vec,0));
	MSET(xprior,k,1,MGET(xsamples,k,1)+VGET(vec,1));
	for(j=0;j<dim;j++){
	  if(MGET(xprior,k,j)<0.0){
	    MSET(xprior,k,j,1.0/gsl_pow_int(10.0,6));
	  }
	}
      }
      margllprop+=trackqprop;
    }

  gsl_matrix_free(xprior);
  gsl_matrix_free(xsamples);
  gsl_matrix_free(disp_mat);
  gsl_matrix_free(disp); gsl_matrix_free(dispInv); gsl_matrix_free(disp2);
  gsl_vector_free(pro); 
  gsl_vector_free(prom);
  gsl_vector_free(obs);
  gsl_vector_free(vec);
  gsl_vector_free(haz);
  
  return margllprop;
}
