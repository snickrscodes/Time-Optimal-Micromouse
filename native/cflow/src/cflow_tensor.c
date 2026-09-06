#include "cflow_internal.h"

static void cheb1_vd(const double *a,int n,double x,double *v,double *d){
    if(n<=0){*v=*d=0;return;} if(n==1){*v=a[0];*d=0;return;}
    double b1=0,b2=0,db1=0,db2=0;
    for(int k=n-1;k>=1;k--){double b0=fma(2*x,b1,a[k])-b2;double db0=2*b1+2*x*db1-db2;b2=b1;b1=b0;db2=db1;db1=db0;}
    *v=fma(x,b1,a[0])-b2;*d=b1+x*db1-db2;
}
static double cheb1(const double *a,int n,double x){double v,d;cheb1_vd(a,n,x,&v,&d);return v;}

static double cheb1_value(const double *a,int n,double x){
    if(n<=0)return 0.0;
    if(n==1)return a[0];
    double b1=0.0,b2=0.0;
    for(int k=n-1;k>=1;k--){
        double b0=fma(2.0*x,b1,a[k])-b2;
        b2=b1;b1=b0;
    }
    return fma(x,b1,a[0])-b2;
}

double cflow_cheb3_value(const double *coef,int n0,int n1,int n2,double x0,double x1,double x2){
    double vi[64],zv[64];
    CFLOW_ASSERT(n0<=64 && n1<=64);
    for(int i=0;i<n0;i++){
        for(int j=0;j<n1;j++)
            zv[j]=cheb1_value(coef+((size_t)i*n1+j)*n2,n2,x2);
        vi[i]=cheb1_value(zv,n1,x1);
    }
    return cheb1_value(vi,n0,x0);
}

void cflow_cheb3(const double *coef,int n0,int n1,int n2,double x0,double x1,double x2,cflow_t3_eval *o){
    double vi[64],dyi[64],dzi[64],zv[64],zd[64];
    for(int i=0;i<n0;i++){
        for(int j=0;j<n1;j++)cheb1_vd(coef+((size_t)i*n1+j)*n2,n2,x2,&zv[j],&zd[j]);
        cheb1_vd(zv,n1,x1,&vi[i],&dyi[i]);dzi[i]=cheb1(zd,n1,x1);
    }
    cheb1_vd(vi,n0,x0,&o->k,&o->da);o->db=cheb1(dyi,n0,x0);o->dc=cheb1(dzi,n0,x0);
}
double cflow_cheb2_value(const double *coef,int n0,int n1,double x0,double x1){
    double vi[64];
    CFLOW_ASSERT(n0<=64);
    for(int i=0;i<n0;i++)vi[i]=cheb1_value(coef+(size_t)i*n1,n1,x1);
    return cheb1_value(vi,n0,x0);
}

void cflow_cheb2(const double *coef,int n0,int n1,double x0,double x1,double *v,double *d0,double *d1){
    double vi[64],di[64];for(int i=0;i<n0;i++)cheb1_vd(coef+(size_t)i*n1,n1,x1,&vi[i],&di[i]);cheb1_vd(vi,n0,x0,v,d0);*d1=cheb1(di,n0,x0);
}

void cflow_cheb2_with_normal(const double *coef,const double *normal,int n0,int n1,double x0,double x1,double *v,double *d0,double *d1,double *dn){
    double vi[64],di[64],ni[64];CFLOW_ASSERT(n0<=64);
    for(int i=0;i<n0;i++){cheb1_vd(coef+(size_t)i*n1,n1,x1,&vi[i],&di[i]);ni[i]=cheb1_value(normal+(size_t)i*n1,n1,x1);}
    cheb1_vd(vi,n0,x0,v,d0);*d1=cheb1(di,n0,x0);*dn=cheb1_value(ni,n0,x0);
}
