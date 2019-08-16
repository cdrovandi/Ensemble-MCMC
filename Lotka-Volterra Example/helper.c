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

#define VSET gsl_vector_set
#define VGET gsl_vector_get
#define MSET gsl_matrix_set
#define MGET gsl_matrix_get

double lnd(double x,double mean,double sd)
{
  /* returns the log of a gaussian density with "mean" and "sd" */
  
  double a;
  a=-0.5*log(2*3.14159)-log(sd)-0.5*((x-mean)*(x-mean)/(sd*sd));
  return(a);
}

void drift(gsl_vector *driftVec,gsl_vector *process,gsl_vector *pxstarE,double diff)
{
  //Calculate Sh(x_s)(t-s) and store in driftVec
  VSET(driftVec,0,diff*(VGET(pxstarE,0)*VGET(process,0)-VGET(pxstarE,1)*VGET(process,0)*VGET(process,1)));
  VSET(driftVec,1,diff*(VGET(pxstarE,1)*VGET(process,0)*VGET(process,1)-VGET(pxstarE,2)*VGET(process,1)));
}

void diffusion(gsl_matrix *disp_mat,gsl_vector *process,gsl_vector *pxstarE,double diff)
{
  //Calculate SH(x_s)S' and store in disp_mat
  MSET(disp_mat,0,0,diff*(VGET(pxstarE,0)*VGET(process,0)+VGET(pxstarE,1)*VGET(process,0)*VGET(process,1)));
  MSET(disp_mat,1,1,diff*(VGET(pxstarE,1)*VGET(process,0)*VGET(process,1)+VGET(pxstarE,2)*VGET(process,1)));
  MSET(disp_mat,0,1,-1.0*diff*(VGET(pxstarE,1)*VGET(process,0)*VGET(process,1)));
  MSET(disp_mat,1,0,MGET(disp_mat,0,1));
}

void hst(gsl_matrix *hst_mat,gsl_vector *process,gsl_vector *pxstarE)
{
  //Calculate HS' and store in hst_mat
  gsl_matrix_set_all(hst_mat,0.0);
  MSET(hst_mat,0,0,VGET(pxstarE,0)*VGET(process,0));
  MSET(hst_mat,1,0,-1.0*VGET(pxstarE,1)*VGET(process,0)*VGET(process,1));
  MSET(hst_mat,1,1,VGET(pxstarE,1)*VGET(process,0)*VGET(process,1));
  MSET(hst_mat,2,1,-1.0*VGET(pxstarE,2)*VGET(process,1));
}

void matmul(int d1,int d2,int d3,gsl_matrix *A,gsl_matrix *B,gsl_matrix *C)
{
  /* multiplies A*B where A is d1*d2 and B is d2*d3 - places result in C */

  int i,j,k;
  
  gsl_matrix_set_all(C,0.0);
  for (i=0;i<d1;i++)
    {
      for(j=0;j<d3;j++)
	{
	  for(k=0;k<d2;k++)
	    {
	      MSET(C,i,j,MGET(C,i,j)+MGET(A,i,k)*MGET(B,k,j));
	    }
	}
    }
}

void matvecmul(int d1,int d2,gsl_matrix *A,gsl_vector *B,gsl_vector *C)
{
  /* multiplies A*B where A is d1*d2 and B is d2*1 - places result in vector C */

  int i,j,k;
  
  gsl_vector_set_all(C,0.0);
  for (i=0;i<d1;i++)
    {
      for(k=0;k<d2;k++)
	{
	  VSET(C,i,VGET(C,i)+MGET(A,i,k)*VGET(B,k));
	}
    }
}

void invert(int d,gsl_matrix *A,gsl_matrix *B)
{ 
  int i;
  double det; 
  gsl_vector *s;
  gsl_vector *work;
  gsl_matrix *temporary;
  gsl_matrix *temporary2; 
  gsl_matrix *V;
  s=gsl_vector_alloc(d);
  work=gsl_vector_alloc(d);
  temporary=gsl_matrix_alloc(d,d);
  temporary2=gsl_matrix_alloc(d,d);
  V=gsl_matrix_alloc(d,d); 
  
  /* Inverts A and places result in B */ 
  if(d==2){
    det = MGET(A,0,0)*MGET(A,1,1)-MGET(A,0,1)*MGET(A,1,0);
    MSET(B,0,0,MGET(A,1,1)/det);
    MSET(B,1,1,MGET(A,0,0)/det);
    MSET(B,0,1,-1.0*MGET(A,0,1)/det);
    MSET(B,1,0,-1.0*MGET(A,1,0)/det);
  }else if(d==1){
    MSET(B,0,0,1.0/(MGET(A,0,0)));
  }else{
  gsl_matrix_memcpy(B,A);
  gsl_linalg_SV_decomp(B,V,s,work);
  
  gsl_matrix_set_all(temporary,0.0);
  gsl_matrix_set_all(temporary2,0.0);
  for(i=0;i<d;i++)
    {
      MSET(temporary,i,i,(1.0/VGET(s,i)));
    }
  gsl_matrix_transpose(B);
  matmul(d,d,d,temporary,B,temporary2);
  matmul(d,d,d,V,temporary2,B);
  }
  
  gsl_vector_free(s);
  gsl_vector_free(work);
  gsl_matrix_free(temporary);
  gsl_matrix_free(temporary2); 
  gsl_matrix_free(V);
}

void invert2by2(gsl_matrix *A,gsl_matrix *B)
{ 
  double det; 
  /* Inverts A and places result in B */ 
  det = MGET(A,0,0)*MGET(A,1,1)-MGET(A,0,1)*MGET(A,1,0);
  MSET(B,0,0,MGET(A,1,1)/det);
  MSET(B,1,1,MGET(A,0,0)/det);
  MSET(B,0,1,-1.0*MGET(A,0,1)/det);
  MSET(B,1,0,-1.0*MGET(A,1,0)/det);
}

void mvn_sample(gsl_vector *vec,gsl_vector *me,gsl_matrix *var,int d,gsl_rng *r)
{
  /* Takes a mean vec, and var matrix, and gives vector of MVN(pxcurr,Var) realisations, pxstar */

  gsl_matrix *disp;  
  gsl_vector *ran; 
  gsl_vector *x; 
  int i,j;
  disp=gsl_matrix_alloc(d,d);  
  ran=gsl_vector_alloc(d);    
  x=gsl_vector_alloc(d);      
  gsl_matrix_memcpy(disp,var);
  gsl_linalg_cholesky_decomp(disp);
  for (i=0;i<d;i++)
    {
      for (j=i+1;j<d;j++)
	{
	  MSET(disp,i,j,0.0);
	}
    }
  gsl_vector_set_all(x,0.0);
  for (j=0;j<d;j++)
    {
      VSET(ran,j,gsl_ran_gaussian(r,1.0));
    }
  for (i=0;i<d;i++)
    {
      for (j=0;j<d;j++)
	{
	  VSET(x,i,VGET(x,i)+MGET(disp,i,j)*VGET(ran,j));
	}
    }
  for(i=0;i<d;i++){
    VSET(vec,i,VGET(me,i)+VGET(x,i)); //add mean
  }
  
  gsl_vector_free(ran);
  gsl_vector_free(x);
  gsl_matrix_free(disp);
}

double lmgpdf(int d,gsl_vector *x,gsl_vector *mu,gsl_matrix *var)
{
  /* Returns the log of a multivariate gaussian pdf with mean vec, mu and var matrix, var */
  /* up to an additive constant */

  int i;
  double det=1.0;
  double s=0.0;
  double ll;
  gsl_matrix *disp; 
  gsl_vector *temp;
 
  disp=gsl_matrix_alloc(d,d);
  temp=gsl_vector_alloc(d);

  gsl_matrix_memcpy(disp,var);
  gsl_linalg_cholesky_decomp(disp);
  
  for (i=0;i<d;i++)
    {
      det=det*MGET(disp,i,i);
      VSET(temp,i,VGET(x,i)-VGET(mu,i));
    }

  gsl_linalg_cholesky_svx(disp,temp);

  for (i=0;i<d;i++)
    {
      s=s+(VGET(temp,i))*(VGET(x,i)-VGET(mu,i));
    }
  ll=-1.0*log(det)-0.5*s;

  gsl_vector_free(temp);
  gsl_matrix_free(disp);

  return(ll);
}
