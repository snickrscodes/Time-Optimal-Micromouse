#include "cflow_internal.h"

static const double *block_ptr(int d){
    switch(d){
        case 0:return cflow_core_c0; case 2:return cflow_core_c2; case 4:return cflow_core_c4; case 6:return cflow_core_c6;
        case 8:return cflow_core_c8; case 10:return cflow_core_c10; case 12:return cflow_core_c12; case 14:return cflow_core_c14;
        case 16:return cflow_core_c16; case 18:return cflow_core_c18; default:return NULL;
    }
}

double cflow_core_eval_compact_value(double p,double r,double c,double s){
    double pp[19],rr[19],cc[19],ss[19]; pp[0]=rr[0]=cc[0]=ss[0]=1.0;
    for(int i=1;i<=18;i++){pp[i]=pp[i-1]*p;rr[i]=rr[i-1]*r;cc[i]=cc[i-1]*c;ss[i]=ss[i-1]*s;}
    double k=0.0,scale=1.0;
    for(int d=0;d<=18;d+=2){
        const double *C=block_ptr(d); if(d)scale*=CFLOW_CORE_RSTAR*CFLOW_CORE_RSTAR; int n=d+1;
        for(int i=0;i<n;i++){
            double w=0.0; const double *row=C+(size_t)i*n;
            for(int j=0;j<n;j++)w=fma(row[j],cc[d-j]*ss[j],w);
            k=fma(scale*pp[d-i]*rr[i],w,k);
        }
    }
    return k;
}

#ifdef CFLOW_CORE_VARIANTS
void cflow_core_eval_compact(double p,double r,double c,double s,cflow_core_eval *o){
    double pp[19],rr[19],cc[19],ss[19]; pp[0]=rr[0]=cc[0]=ss[0]=1.0;
    for(int i=1;i<=18;i++){pp[i]=pp[i-1]*p;rr[i]=rr[i-1]*r;cc[i]=cc[i-1]*c;ss[i]=ss[i-1]*s;}
    double k=0,kp=0,kr=0,kth=0,scale=1.0;
    for(int d=0;d<=18;d+=2){
        const double *C=block_ptr(d); if(d)scale*=CFLOW_CORE_RSTAR*CFLOW_CORE_RSTAR; int n=d+1;
        for(int i=0;i<n;i++){
            const double *row=C+(size_t)i*n; double w=0,wth=0;
            for(int j=0;j<n;j++){
                double a=row[j],v=cc[d-j]*ss[j],vt=0.0; w=fma(a,v,w);
                if(d-j)vt-=(d-j)*cc[d-j-1]*ss[j+1];
                if(j)vt+=j*cc[d-j+1]*ss[j-1];
                wth=fma(a,vt,wth);
            }
            double u=pp[d-i]*rr[i],up=(d-i)?(d-i)*pp[d-i-1]*rr[i]:0.0,ur=i?i*pp[d-i]*rr[i-1]:0.0;
            k=fma(scale*u,w,k);kp=fma(scale*up,w,kp);kr=fma(scale*ur,w,kr);kth=fma(scale*u,wth,kth);
        }
    }
    o->k=k;o->kp=kp;o->kr=kr;o->kc=kth;o->ks=0.0;
}
#endif

/* Degree-block value+gradient evaluator used by the release Jacobian path. */
static void partial_block(int d,const double *C,double scale,const double *pp,const double *rr,const double *cc,const double *ss,double *k,double *kp,double *kr,double *kth){
    int n=d+1;
    for(int i=0;i<n;i++){
        const double *row=C+(size_t)i*n; double w=0,wth=0;
        for(int j=0;j<n;j++){
            double a=row[j],v=cc[d-j]*ss[j],vt=0.0; w=fma(a,v,w);
            if(d-j) vt-=(d-j)*cc[d-j-1]*ss[j+1];
            if(j) vt+=j*cc[d-j+1]*ss[j-1];
            wth=fma(a,vt,wth);
        }
        double u=pp[d-i]*rr[i],up=(d-i)?(d-i)*pp[d-i-1]*rr[i]:0.0,ur=i?i*pp[d-i]*rr[i-1]:0.0;
        *k=fma(scale*u,w,*k);*kp=fma(scale*up,w,*kp);*kr=fma(scale*ur,w,*kr);*kth=fma(scale*u,wth,*kth);
    }
}
void cflow_core_eval_partial(double p,double r,double c,double s,cflow_core_eval *o){
    double pp[19],rr[19],cc[20],ss[20];pp[0]=rr[0]=cc[0]=ss[0]=1;
    for(int i=1;i<=19;i++){if(i<=18){pp[i]=pp[i-1]*p;rr[i]=rr[i-1]*r;}cc[i]=cc[i-1]*c;ss[i]=ss[i-1]*s;}
    const double sc2=CFLOW_CORE_RSTAR*CFLOW_CORE_RSTAR;
    double k=0,kp=0,kr=0,kth=0,sc=1; partial_block(0,cflow_core_c0,sc,pp,rr,cc,ss,&k,&kp,&kr,&kth);
    sc*=sc2;partial_block(2,cflow_core_c2,sc,pp,rr,cc,ss,&k,&kp,&kr,&kth);sc*=sc2;partial_block(4,cflow_core_c4,sc,pp,rr,cc,ss,&k,&kp,&kr,&kth);
    sc*=sc2;partial_block(6,cflow_core_c6,sc,pp,rr,cc,ss,&k,&kp,&kr,&kth);sc*=sc2;partial_block(8,cflow_core_c8,sc,pp,rr,cc,ss,&k,&kp,&kr,&kth);
    sc*=sc2;partial_block(10,cflow_core_c10,sc,pp,rr,cc,ss,&k,&kp,&kr,&kth);sc*=sc2;partial_block(12,cflow_core_c12,sc,pp,rr,cc,ss,&k,&kp,&kr,&kth);
    sc*=sc2;partial_block(14,cflow_core_c14,sc,pp,rr,cc,ss,&k,&kp,&kr,&kth);sc*=sc2;partial_block(16,cflow_core_c16,sc,pp,rr,cc,ss,&k,&kp,&kr,&kth);
    sc*=sc2;partial_block(18,cflow_core_c18,sc,pp,rr,cc,ss,&k,&kp,&kr,&kth);o->k=k;o->kp=kp;o->kr=kr;o->kc=kth;o->ks=0;
}
#ifdef CFLOW_CORE_VARIANTS
double cflow_core_eval_partial_value(double p,double r,double c,double s){
    return cflow_core_eval_compact_value(p,r,c,s);
}
#endif

#ifndef CFLOW_HAVE_UNROLLED
void cflow_core_eval_unrolled(double p,double r,double c,double s,cflow_core_eval *o){ cflow_core_eval_partial(p,r,c,s,o); }
#endif

int cflow_core_legal(double x,double q,double b,double h){
    double S=hypot(x,2.0*h); if(S==0.0) return 1;
    double p=fabs(q)*S/CFLOW_CORE_RSTAR; double r=fabs(b*h)*S/CFLOW_CORE_RSTAR;
    return p<=1.0 && r<=1.0;
}

double cflow_core_cap(double x,double q,double b,double remaining){
    if(remaining<=0.0) return 0.0;
    if(fabs(q*x)>=CFLOW_Z_CORE_SWITCH) return 0.0;
    double h=remaining;
    if(cflow_core_legal(x,q,b,h*CFLOW_CORE_SAFETY)) return h;
    /* Scale-safe monotone halving fallback. It cannot extrapolate and avoids overflow-prone quartics. */
    for(int i=0;i<1100;i++){ h*=0.5; if(h==0.0) return 0.0; if(cflow_core_legal(x,q,b,h/CFLOW_CORE_SAFETY)) return h*CFLOW_CORE_SAFETY; }
    return 0.0;
}

void cflow_core_local(double x,double q,double b,double h,int want_jac,cflow_local_jac *o){
    CFLOW_ASSERT(cflow_core_legal(x,q,b,h));
    double S=hypot(x,2.0*h); if(S==0.0){ o->x=x;o->dx=1;o->dq=o->db=0;return; }
    double c=x/S, s=2.0*h/S, p=q*S/CFLOW_CORE_RSTAR, r=b*h*S/CFLOW_CORE_RSTAR;
    cflow_core_eval e; if(!want_jac){ e.k=cflow_core_eval_compact_value(p,r,c,s); e.kp=e.kr=e.kc=e.ks=0.0; } else {
cflow_core_eval_partial(p,r,c,s,&e);
    }
    o->x=x+2.0*h*e.k;
    if(!want_jac){o->dx=o->dq=o->db=NAN;return;}
    double invR=1.0/CFLOW_CORE_RSTAR, invS=1.0/S;
    double dpdx=q*c*invR, dpdq=S*invR, dpdh=2.0*q*s*invR;
    double drdx=b*h*c*invR, drdb=h*S*invR, drdh=b*S*(1.0+s*s)*invR;
    double dthdx=-s*invS, dthdh=2.0*c*invS;
    double kx=e.kp*dpdx+e.kr*drdx+e.kc*dthdx;
    double kq=e.kp*dpdq;
    double kb=e.kr*drdb;
    double kh=e.kp*dpdh+e.kr*drdh+e.kc*dthdh;
    o->dx=1.0+2.0*h*kx; o->dq=2.0*h*kq; o->db=2.0*h*kb;
    (void)kh; /* final-time derivative is evaluated exactly from the endpoint by the dispatcher. */
}

double cflow_core_dx_defect(double x,double q,double b,double h){
    double S=hypot(x,2.0*h); if(S==0.0) return 0.0;
    double c=x/S, s=2.0*h/S, p=q*S/CFLOW_CORE_RSTAR, r=b*h*S/CFLOW_CORE_RSTAR;
    cflow_core_eval e; cflow_core_eval_partial(p,r,c,s,&e);
    double invR=1.0/CFLOW_CORE_RSTAR, invS=1.0/S;
    double dpdx=q*c*invR, drdx=b*h*c*invR, dthdx=-s*invS;
    double kx=e.kp*dpdx+e.kr*drdx+e.kc*dthdx;
    return 2.0*h*kx;
}
