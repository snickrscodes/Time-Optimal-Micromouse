#include "cflow_internal.h"
#include "integral_tables.h"
#include "highz_face_tables.h"
#include "quadrature.h"
#include <string.h>

typedef struct { double v,d0,d1,d2,d22; } ituck;
static void ibasis(const double *U,int n,int r,double x,double *a,double *da,double *d2a){
    double T[48],D[48],D2[48];
    T[0]=1.0;D[0]=D2[0]=0.0;
    if(n>1){T[1]=x;D[1]=1.0;D2[1]=0.0;}
    for(int k=2;k<n;k++){T[k]=2.0*x*T[k-1]-T[k-2];D[k]=2.0*T[k-1]+2.0*x*D[k-1]-D[k-2];D2[k]=4.0*D[k-1]+2.0*x*D2[k-1]-D2[k-2];}
    for(int j=0;j<r;j++){double v=0,d=0,d2=0;for(int k=0;k<n;k++){double u=U[(size_t)k*r+j];v=fma(u,T[k],v);d=fma(u,D[k],d);d2=fma(u,D2[k],d2);}a[j]=v;da[j]=d;d2a[j]=d2;}
}
static void itucker(const double *G,const double *U0,const double *U1,const double *U2,int n0,int n1,int n2,int r0,int r1,int r2,double x0,double x1,double x2,ituck *o){
    double a[16],da[16],d2a[16],bb[16],db[16],d2b[16],c[16],dc[16],d2c[16];
    double z[256],z2[256],z22[256],y[16],y1[16],y2[16],y22[16];
    ibasis(U0,n0,r0,x0,a,da,d2a);ibasis(U1,n1,r1,x1,bb,db,d2b);ibasis(U2,n2,r2,x2,c,dc,d2c);
    for(int i=0;i<r0;i++)for(int j=0;j<r1;j++){
        const double *g=G+((size_t)i*r1+j)*r2;double v=0,d=0,d2=0;
        for(int k=0;k<r2;k++){v=fma(g[k],c[k],v);d=fma(g[k],dc[k],d);d2=fma(g[k],d2c[k],d2);}z[i*r1+j]=v;z2[i*r1+j]=d;z22[i*r1+j]=d2;
    }
    for(int i=0;i<r0;i++){double v=0,d1=0,d2=0,d22=0;for(int j=0;j<r1;j++){double zz=z[i*r1+j];v=fma(zz,bb[j],v);d1=fma(zz,db[j],d1);d2=fma(z2[i*r1+j],bb[j],d2);d22=fma(z22[i*r1+j],bb[j],d22);}y[i]=v;y1[i]=d1;y2[i]=d2;y22[i]=d22;}
    o->v=o->d0=o->d1=o->d2=o->d22=0.0;
    for(int i=0;i<r0;i++){o->v=fma(y[i],a[i],o->v);o->d0=fma(y[i],da[i],o->d0);o->d1=fma(y1[i],a[i],o->d1);o->d2=fma(y2[i],a[i],o->d2);o->d22=fma(y22[i],a[i],o->d22);}
}


static void ibasis_value(const double *U,int n,int rr,double x,double *a){
    double T[48];T[0]=1.0;if(n>1)T[1]=x;for(int k=2;k<n;k++)T[k]=2.0*x*T[k-1]-T[k-2];
    for(int j=0;j<rr;j++){double v=0.0;for(int k=0;k<n;k++)v=fma(U[(size_t)k*rr+j],T[k],v);a[j]=v;}
}
static double itucker_value(const double *G,const double *U0,const double *U1,const double *U2,int n0,int n1,int n2,int r0,int r1,int r2,double x0,double x1,double x2){
    double a[16],bb[16],c[16],z[256],y[16];ibasis_value(U0,n0,r0,x0,a);ibasis_value(U1,n1,r1,x1,bb);ibasis_value(U2,n2,r2,x2,c);
    for(int i=0;i<r0;i++)for(int j=0;j<r1;j++){const double*g=G+((size_t)i*r1+j)*r2;double v=0.0;for(int k=0;k<r2;k++)v=fma(g[k],c[k],v);z[i*r1+j]=v;}
    for(int i=0;i<r0;i++){double v=0.0;for(int j=0;j<r1;j++)v=fma(z[i*r1+j],bb[j],v);y[i]=v;}
    double v=0.0;for(int i=0;i<r0;i++)v=fma(y[i],a[i],v);return v;
}

static double free_j(double x,double h){
    double a=sqrt(x),z=sqrt(x+2.0*h);return (2.0*h)/(z+a); /* z-a, stable */
}

void cflow_integral_core_local(double x,double q,double b,double h,cflow_integral_local *o){
    double S=hypot(x,2.0*h);if(S==0.0){o->j=o->jq=o->jb=o->jx=o->regx=0.0;return;}
    double rootS=sqrt(S),c=x/S,s=2.0*h/S,P=q*S,Q=b*h*S;ituck e;double R,RP,RQ,Rx;
    double th=atan2(s,c);
    if(th<=1.15){
        double zd=2.0*th/1.25-1.0;
        itucker(cflow_int_core_angle_g,cflow_int_core_angle_u0,cflow_int_core_angle_u1,cflow_int_core_angle_u2,
                CFLOW_INT_CORE_ANGLE_N0,CFLOW_INT_CORE_ANGLE_N1,CFLOW_INT_CORE_ANGLE_N2,CFLOW_INT_CORE_ANGLE_R0,CFLOW_INT_CORE_ANGLE_R1,CFLOW_INT_CORE_ANGLE_R2,
                P/.2,Q/.2,zd,&e);
        R=e.v;RP=e.d0/.2;RQ=e.d1/.2;double Rd=e.d2*(2.0/1.25);
        Rx=c*R/(2.0*rootS)+rootS*(RP*q*c+RQ*b*h*c-Rd*s/S);
    }else{
        double am=sqrt(.5),alpha=sqrt(fmax(0.0,c)),zd=2.0*alpha/am-1.0;
        itucker(cflow_int_core_alpha_g,cflow_int_core_alpha_u0,cflow_int_core_alpha_u1,cflow_int_core_alpha_u2,
                CFLOW_INT_CORE_ALPHA_N0,CFLOW_INT_CORE_ALPHA_N1,CFLOW_INT_CORE_ALPHA_N2,CFLOW_INT_CORE_ALPHA_R0,CFLOW_INT_CORE_ALPHA_R1,CFLOW_INT_CORE_ALPHA_R2,
                P/.2,Q/.2,zd,&e);
        R=e.v;RP=e.d0/.2;RQ=e.d1/.2;double za=2.0/am,Ra=e.d2*za;
        double Rdc;
        if(alpha>1e-7) Rdc=Ra/(2.0*alpha);
        else Rdc=0.5*e.d22*za*za; /* dR/dc at c=alpha^2, alpha->0 */
        Rx=c*R/(2.0*rootS)+rootS*(RP*q*c+RQ*b*h*c+Rdc*s*s/S);
    }
    double fj=free_j(x,h);o->j=fj+rootS*R;
    if(th>1.15 && sqrt(fmax(0.0,c))<.08){
        double alpha=sqrt(fmax(0.0,c)),zd=2.0*alpha/.08-1.0;ituck g;
        itucker(cflow_int_core_regx_near_g,cflow_int_core_regx_near_u0,cflow_int_core_regx_near_u1,cflow_int_core_regx_near_u2,CFLOW_INT_CORE_REGX_NEAR_N0,CFLOW_INT_CORE_REGX_NEAR_N1,CFLOW_INT_CORE_REGX_NEAR_N2,CFLOW_INT_CORE_REGX_NEAR_R0,CFLOW_INT_CORE_REGX_NEAR_R1,CFLOW_INT_CORE_REGX_NEAR_R2,P/.2,Q/.2,zd,&g);
        o->regx=g.v/rootS;
    }else o->regx=0.5/sqrt(x+2.0*h)+Rx;
    o->jx=x>0.0?(-0.5/sqrt(x)+o->regx):-INFINITY;
    o->jq=rootS*RP*S;o->jb=rootS*RQ*h*S;
}

typedef struct {
    double zlo,zhi,cmax,smax,tlo,thi;
    int n0,n1,n2;
    const double *cface_p,*cface_m,*cface_d_p,*cface_d_m;
    const double *sface_p,*sface_m,*sface_d_p,*sface_d_m;
} ihzspec;
static const ihzspec IHP[3]={
 {0.15,0.80,.010,.12,0.15056827277668602,0.9272952180016123,
  CFLOW_INT_HZ_MID_N0,CFLOW_INT_HZ_MID_N1,CFLOW_INT_HZ_MID_N2,
  cflow_int_hz_mid_cface_p,cflow_int_hz_mid_cface_m,cflow_int_hz_mid_cface_d_p,cflow_int_hz_mid_cface_d_m,
  cflow_int_hz_mid_sface_p,cflow_int_hz_mid_sface_m,cflow_int_hz_mid_sface_d_p,cflow_int_hz_mid_sface_d_m},
 {0.75,.94,.004,.045,0.848062078981481,1.2226303055219356,
  CFLOW_INT_HZ_UPPER_N0,CFLOW_INT_HZ_UPPER_N1,CFLOW_INT_HZ_UPPER_N2,
  cflow_int_hz_upper_cface_p,cflow_int_hz_upper_cface_m,cflow_int_hz_upper_cface_d_p,cflow_int_hz_upper_cface_d_m,
  cflow_int_hz_upper_sface_p,cflow_int_hz_upper_sface_m,cflow_int_hz_upper_sface_d_p,cflow_int_hz_upper_sface_d_m},
 {.92,.9800665778412416,.001,.010,1.1680804852142352,1.3707963267948966,
  CFLOW_INT_HZ_NEAR_N0,CFLOW_INT_HZ_NEAR_N1,CFLOW_INT_HZ_NEAR_N2,
  NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL}
};
static void ihztuck(int panel,double a,double c,double s,ituck *e){
 if(panel==0)itucker(cflow_int_hz_mid_g,cflow_int_hz_mid_u0,cflow_int_hz_mid_u1,cflow_int_hz_mid_u2,CFLOW_INT_HZ_MID_N0,CFLOW_INT_HZ_MID_N1,CFLOW_INT_HZ_MID_N2,CFLOW_INT_HZ_MID_R0,CFLOW_INT_HZ_MID_R1,CFLOW_INT_HZ_MID_R2,a,c,s,e);
 else if(panel==1)itucker(cflow_int_hz_upper_g,cflow_int_hz_upper_u0,cflow_int_hz_upper_u1,cflow_int_hz_upper_u2,CFLOW_INT_HZ_UPPER_N0,CFLOW_INT_HZ_UPPER_N1,CFLOW_INT_HZ_UPPER_N2,CFLOW_INT_HZ_UPPER_R0,CFLOW_INT_HZ_UPPER_R1,CFLOW_INT_HZ_UPPER_R2,a,c,s,e);
 else itucker(cflow_int_hz_near_g,cflow_int_hz_near_u0,cflow_int_hz_near_u1,cflow_int_hz_near_u2,CFLOW_INT_HZ_NEAR_N0,CFLOW_INT_HZ_NEAR_N1,CFLOW_INT_HZ_NEAR_N2,CFLOW_INT_HZ_NEAR_R0,CFLOW_INT_HZ_NEAR_R1,CFLOW_INT_HZ_NEAR_R2,a,c,s,e);
}
static int ihz_is_c_face(cflow_highz_face face){return face==CFLOW_HZ_FACE_C_POS||face==CFLOW_HZ_FACE_C_NEG;}
static int ihz_is_s_face(cflow_highz_face face){return face==CFLOW_HZ_FACE_S_POS||face==CFLOW_HZ_FACE_S_NEG;}
static const double *ihz_face_value(const ihzspec *p,cflow_highz_face face){
    if(face==CFLOW_HZ_FACE_C_POS)return p->cface_p;
    if(face==CFLOW_HZ_FACE_C_NEG)return p->cface_m;
    if(face==CFLOW_HZ_FACE_S_POS)return p->sface_p;
    if(face==CFLOW_HZ_FACE_S_NEG)return p->sface_m;
    return NULL;
}
static const double *ihz_face_normal(const ihzspec *p,cflow_highz_face face){
    if(face==CFLOW_HZ_FACE_C_POS)return p->cface_d_p;
    if(face==CFLOW_HZ_FACE_C_NEG)return p->cface_d_m;
    if(face==CFLOW_HZ_FACE_S_POS)return p->sface_d_p;
    if(face==CFLOW_HZ_FACE_S_NEG)return p->sface_d_m;
    return NULL;
}
static void ibasis_first(const double *U,int n,int r,double x,double *a,double *da){
 double T[48],D[48];T[0]=1.0;D[0]=0.0;if(n>1){T[1]=x;D[1]=1.0;}
 for(int k=2;k<n;k++){T[k]=2.0*x*T[k-1]-T[k-2];D[k]=2.0*T[k-1]+2.0*x*D[k-1]-D[k-2];}
 for(int j=0;j<r;j++){double v=0.0,d=0.0;for(int k=0;k<n;k++){double u=U[(size_t)k*r+j];v=fma(u,T[k],v);d=fma(u,D[k],d);}a[j]=v;da[j]=d;}
}
static const double *near_face_core(cflow_highz_face face){
 switch(face){
  case CFLOW_HZ_FACE_C_POS:return cflow_int_hz_near_cface_core_p;
  case CFLOW_HZ_FACE_C_NEG:return cflow_int_hz_near_cface_core_m;
  case CFLOW_HZ_FACE_S_POS:return cflow_int_hz_near_sface_core_p;
  case CFLOW_HZ_FACE_S_NEG:return cflow_int_hz_near_sface_core_m;
  default:return NULL;
 }
}
static const double *near_face_d_core(cflow_highz_face face){
 switch(face){
  case CFLOW_HZ_FACE_C_POS:return cflow_int_hz_near_cface_d_core_p;
  case CFLOW_HZ_FACE_C_NEG:return cflow_int_hz_near_cface_d_core_m;
  case CFLOW_HZ_FACE_S_POS:return cflow_int_hz_near_sface_d_core_p;
  case CFLOW_HZ_FACE_S_NEG:return cflow_int_hz_near_sface_d_core_m;
  default:return NULL;
 }
}
static void ihz_near_face_eval(cflow_highz_face face,double a0,double c0,double s0,ituck *e){
 const int cface=ihz_is_c_face(face);const double *U=cface?cflow_int_hz_near_u2:cflow_int_hz_near_u1;
 const int n=cface?CFLOW_INT_HZ_NEAR_N2:CFLOW_INT_HZ_NEAR_N1,r=cface?CFLOW_INT_HZ_NEAR_R2:CFLOW_INT_HZ_NEAR_R1;
 const double y0=cface?s0:c0;double a[CFLOW_INT_HZ_NEAR_R0],da[CFLOW_INT_HZ_NEAR_R0],y[CFLOW_INT_HZ_NEAR_R2],dy[CFLOW_INT_HZ_NEAR_R2];
 ibasis_first(cflow_int_hz_near_u0,CFLOW_INT_HZ_NEAR_N0,CFLOW_INT_HZ_NEAR_R0,a0,a,da);ibasis_first(U,n,r,y0,y,dy);
 const double *H=near_face_core(face),*Hd=near_face_d_core(face);double tang=0.0,norm=0.0;e->v=e->d0=e->d1=e->d2=e->d22=0.0;
 for(int i=0;i<CFLOW_INT_HZ_NEAR_R0;i++){double z=0.0,zn=0.0,zt=0.0;for(int k=0;k<r;k++){double h=H[(size_t)i*r+k];z=fma(h,y[k],z);zn=fma(Hd[(size_t)i*r+k],y[k],zn);zt=fma(h,dy[k],zt);}e->v=fma(z,a[i],e->v);e->d0=fma(z,da[i],e->d0);norm=fma(zn,a[i],norm);tang=fma(zt,a[i],tang);}
 if(cface){e->d1=norm;e->d2=tang;}else{e->d1=tang;e->d2=norm;}
}
static double ihz_near_face_value(cflow_highz_face face,double a0,double c0,double s0){
 const int cface=ihz_is_c_face(face);const double *U=cface?cflow_int_hz_near_u2:cflow_int_hz_near_u1;
 const int n=cface?CFLOW_INT_HZ_NEAR_N2:CFLOW_INT_HZ_NEAR_N1,r=cface?CFLOW_INT_HZ_NEAR_R2:CFLOW_INT_HZ_NEAR_R1;const double y0=cface?s0:c0;
 double a[CFLOW_INT_HZ_NEAR_R0],y[CFLOW_INT_HZ_NEAR_R2];ibasis_value(cflow_int_hz_near_u0,CFLOW_INT_HZ_NEAR_N0,CFLOW_INT_HZ_NEAR_R0,a0,a);ibasis_value(U,n,r,y0,y);
 const double *H=near_face_core(face);double v=0.0;for(int i=0;i<CFLOW_INT_HZ_NEAR_R0;i++){double z=0.0;for(int k=0;k<r;k++)z=fma(H[(size_t)i*r+k],y[k],z);v=fma(z,a[i],v);}return v;
}
static void ihz_eval(int panel,cflow_highz_face face,double a,double c,double s,ituck *e){
 const ihzspec *p=&IHP[panel];e->d22=0.0;
 if(panel==2 && face!=CFLOW_HZ_FACE_NONE){ihz_near_face_eval(face,a,c,s,e);return;}
 if(ihz_is_c_face(face)){
   cflow_cheb2_with_normal(ihz_face_value(p,face),ihz_face_normal(p,face),p->n0,p->n2,a,s,&e->v,&e->d0,&e->d2,&e->d1);return;
 }
 if(ihz_is_s_face(face)){
   cflow_cheb2_with_normal(ihz_face_value(p,face),ihz_face_normal(p,face),p->n0,p->n1,a,c,&e->v,&e->d0,&e->d1,&e->d2);return;
 }
 ihztuck(panel,a,c,s,e);
}
static double ihz_value(int panel,cflow_highz_face face,double a,double c,double s){
 const ihzspec *p=&IHP[panel];
 if(panel==2 && face!=CFLOW_HZ_FACE_NONE)return ihz_near_face_value(face,a,c,s);
 if(ihz_is_c_face(face))return cflow_cheb2_value(ihz_face_value(p,face),p->n0,p->n2,a,s);
 if(ihz_is_s_face(face))return cflow_cheb2_value(ihz_face_value(p,face),p->n0,p->n1,a,c);
 if(panel==0)return itucker_value(cflow_int_hz_mid_g,cflow_int_hz_mid_u0,cflow_int_hz_mid_u1,cflow_int_hz_mid_u2,CFLOW_INT_HZ_MID_N0,CFLOW_INT_HZ_MID_N1,CFLOW_INT_HZ_MID_N2,CFLOW_INT_HZ_MID_R0,CFLOW_INT_HZ_MID_R1,CFLOW_INT_HZ_MID_R2,a,c,s);
 if(panel==1)return itucker_value(cflow_int_hz_upper_g,cflow_int_hz_upper_u0,cflow_int_hz_upper_u1,cflow_int_hz_upper_u2,CFLOW_INT_HZ_UPPER_N0,CFLOW_INT_HZ_UPPER_N1,CFLOW_INT_HZ_UPPER_N2,CFLOW_INT_HZ_UPPER_R0,CFLOW_INT_HZ_UPPER_R1,CFLOW_INT_HZ_UPPER_R2,a,c,s);
 return itucker_value(cflow_int_hz_near_g,cflow_int_hz_near_u0,cflow_int_hz_near_u1,cflow_int_hz_near_u2,CFLOW_INT_HZ_NEAR_N0,CFLOW_INT_HZ_NEAR_N1,CFLOW_INT_HZ_NEAR_N2,CFLOW_INT_HZ_NEAR_R0,CFLOW_INT_HZ_NEAR_R1,CFLOW_INT_HZ_NEAR_R2,a,c,s);
}
void cflow_integral_highz_local(double x,double q,double b,double h,int panel,cflow_highz_face face,cflow_integral_local *o){
    const ihzspec *p=&IHP[panel];double z=q*x,sig=z>=0?1.0:-1.0,az=fabs(z),ct=sqrt(fmax(0.0,1.0-az*az)),th=asin(az),C=sig*b*h*x,sv=2.0*h/x;
    double xt=(2.0*th-p->tlo-p->thi)/(p->thi-p->tlo),xc=C/p->cmax,xs=sv/p->smax;ituck e;ihz_eval(panel,face,xt,xc,xs,&e);
    double Rt=e.d0*2.0/(p->thi-p->tlo),Rc=e.d1/p->cmax,Rs=e.d2/p->smax,rootx=sqrt(x),R=e.v;
    double Rx=R/(2.0*rootx)+rootx*(Rt*(sig*q/ct)+Rc*(sig*b*h)+Rs*(-sv/x));
    o->j=free_j(x,h)+rootx*R;o->regx=0.5/sqrt(x+2.0*h)+Rx;o->jx=-0.5/rootx+o->regx;
    o->jq=rootx*Rt*(sig*x/ct);o->jb=rootx*Rc*(sig*h*x);
}

/* Free-root fixed Gaussian fallback. Every node is a shorter substep of the
   already legal local chart; core/high-z/far-field domains are monotone under
   this shortening, so the same local evaluator remains in-domain. */
static void residual_from_flow(int kind,int panel,double x,double q,double b,double t,double *r,double *rx,double *rq,double *rb){
    cflow_local_jac f;cflow_eval_local(kind,panel,CFLOW_HZ_FACE_NONE,x,q,b,t,1,&f);
    double X=f.x,F=x+2.0*t,a=sqrt(X),ff=sqrt(F),del=F-X;
    *r=del==0.0?0.0:del/(a*ff*(a+ff));
    *rq=-0.5*f.dq/(X*a);*rb=-0.5*f.db/(X*a);
    double dm1;
    if(kind==0)dm1=cflow_core_dx_defect(x,q,b,t);
    else if(kind==1)dm1=cflow_highz_dx_defect(x,q,b,t,panel);
    else if(kind==3)dm1=cflow_farfield_dx_defect(x,q,b,t);
    else dm1=f.dx-1.0;
    double t2=del==0.0?0.0:del*(F+ff*a+X)/((ff+a)*(X*a)*(F*ff));
    *rx=-0.5*(dm1/(X*a)+t2);
}

static void fallback_quad_n(int kind,int panel,double x,double q,double b,double h,int n,cflow_integral_local *o){
    const double *gn=n==12?cflow_gl12_x:cflow_gl16_x;
    const double *gw=n==12?cflow_gl12_w:cflow_gl16_w;
    double z0=sqrt(x),z1=sqrt(x+2.0*h),dz=z1-z0,rr=0,rx=0,rq=0,rb=0;
    for(int i=0;i<n;i++){
        double y=gn[i],z=z0+dz*y,t=.5*dz*y*(2*z0+dz*y),a,bx,c,d;
        residual_from_flow(kind,panel,x,q,b,t,&a,&bx,&c,&d);
        double f=dz*z*gw[i];rr=fma(f,a,rr);rx=fma(f,bx,rx);rq=fma(f,c,rq);rb=fma(f,d,rb);
    }
    o->j=dz+rr;o->regx=.5/z1+rx;o->jx=z0>0?-.5/z0+o->regx:-INFINITY;o->jq=rq;o->jb=rb;
}

int cflow_integral_farfield_local(double x,double q,double b,double h,cflow_integral_local *o){
    if(q==0.0)return 0;
    double Q=fabs(q),sig=copysign(1.0,q),xb=Q*x,th=asin(fmin(1.0,xb)),eps=sig*b/(Q*Q),sv=sig*h*(2.0*q+b*h);
    if(th<.15){fallback_quad_n(3,-1,x,q,b,h,16,o);return 1;}
    if(th>1.05||fabs(eps)>.05||sv<0||sv>.18)return 0;
    ituck e;itucker(cflow_int_far_theta_g,cflow_int_far_theta_u0,cflow_int_far_theta_u1,cflow_int_far_theta_u2,
        CFLOW_INT_FAR_THETA_N0,CFLOW_INT_FAR_THETA_N1,CFLOW_INT_FAR_THETA_N2,
        CFLOW_INT_FAR_THETA_R0,CFLOW_INT_FAR_THETA_R1,CFLOW_INT_FAR_THETA_R2,
        eps/.05,(2.0*th-.15-1.05)/(1.05-.15),2.0*sv/.18-1.0,&e);
    double Re=e.d0/.05,Rt=e.d1*2.0/(1.05-.15),Rs=e.d2*2.0/.18,R=e.v,rootQ=sqrt(Q),ct=cos(th);
    double Rphys=R/rootQ,Rx=rootQ*Rt/ct;
    double eq=-2.0*eps*sig/Q,thq=sig*x/ct,sq=2.0*sig*h;
    double Rq=-.5*sig*R/(Q*rootQ)+(Re*eq+Rt*thq+Rs*sq)/rootQ;
    double Rb=(Re*sig/(Q*Q)+Rs*sig*h*h)/rootQ;
    o->j=free_j(x,h)+Rphys;o->regx=.5/sqrt(x+2*h)+Rx;
    o->jx=x>0?-.5/sqrt(x)+o->regx:-INFINITY;o->jq=Rq;o->jb=Rb;return 1;
}


static double integral_core_value(double x,double q,double b,double h){
    double S=hypot(x,2.0*h);if(S==0.0)return 0.0;double rootS=sqrt(S),c=x/S,ss=2.0*h/S,P=q*S,Q=b*h*S,R;double th=atan2(ss,c);
    if(th<=1.15){double zd=2.0*th/1.25-1.0;R=itucker_value(cflow_int_core_angle_g,cflow_int_core_angle_u0,cflow_int_core_angle_u1,cflow_int_core_angle_u2,CFLOW_INT_CORE_ANGLE_N0,CFLOW_INT_CORE_ANGLE_N1,CFLOW_INT_CORE_ANGLE_N2,CFLOW_INT_CORE_ANGLE_R0,CFLOW_INT_CORE_ANGLE_R1,CFLOW_INT_CORE_ANGLE_R2,P/.2,Q/.2,zd);}
    else{double am=sqrt(.5),alpha=sqrt(fmax(0.0,c)),zd=2.0*alpha/am-1.0;R=itucker_value(cflow_int_core_alpha_g,cflow_int_core_alpha_u0,cflow_int_core_alpha_u1,cflow_int_core_alpha_u2,CFLOW_INT_CORE_ALPHA_N0,CFLOW_INT_CORE_ALPHA_N1,CFLOW_INT_CORE_ALPHA_N2,CFLOW_INT_CORE_ALPHA_R0,CFLOW_INT_CORE_ALPHA_R1,CFLOW_INT_CORE_ALPHA_R2,P/.2,Q/.2,zd);}
    return free_j(x,h)+rootS*R;
}
static double integral_highz_value(double x,double q,double b,double h,int panel,cflow_highz_face face){
    const ihzspec*p=&IHP[panel];double z=q*x,sig=z>=0?1.0:-1.0,az=fabs(z),th=asin(az),C=sig*b*h*x,sv=2.0*h/x;
    double xt=(2.0*th-p->tlo-p->thi)/(p->thi-p->tlo),xc=C/p->cmax,xs=sv/p->smax;
    double R=ihz_value(panel,face,xt,xc,xs);
    return free_j(x,h)+sqrt(x)*R;
}
static double integral_far_value(double x,double q,double b,double h,int *ok){
    if(q==0.0){*ok=0;return NAN;}double Q=fabs(q),sig=copysign(1.0,q),xb=Q*x,th=asin(fmin(1.0,xb)),eps=sig*b/(Q*Q),sv=sig*h*(2.0*q+b*h);if(th<.15||th>1.05||fabs(eps)>.05||sv<0||sv>.18){*ok=0;return NAN;}
    double R=itucker_value(cflow_int_far_theta_g,cflow_int_far_theta_u0,cflow_int_far_theta_u1,cflow_int_far_theta_u2,CFLOW_INT_FAR_THETA_N0,CFLOW_INT_FAR_THETA_N1,CFLOW_INT_FAR_THETA_N2,CFLOW_INT_FAR_THETA_R0,CFLOW_INT_FAR_THETA_R1,CFLOW_INT_FAR_THETA_R2,eps/.05,(2.0*th-.15-1.05)/(1.05-.15),2.0*sv/.18-1.0);*ok=1;return free_j(x,h)+R/sqrt(Q);
}
static int eval_segment_value(int kind,int panel,cflow_highz_face face,double x,double q,double b,double h,cflow_local_jac*f,double*j){
    int ok=1;if(kind==0)*j=integral_core_value(x,q,b,h);else if(kind==1)*j=integral_highz_value(x,q,b,h,panel,face);else if(kind==2){if(!cflow_integral_terminal_value_local(x,q,b,h,j))return 0;}else if(kind==3)*j=integral_far_value(x,q,b,h,&ok);else ok=0;
    if(!ok){cflow_integral_local ij;fallback_quad_n(kind,panel,x,q,b,h,12,&ij);*j=ij.j;}
    cflow_eval_local(kind,panel,face,x,q,b,h,0,f);return isfinite(*j)&&isfinite(f->x);
}

static int eval_segment_all(int kind,int panel,cflow_highz_face face,double x,double q,double b,double h,cflow_local_jac *f,cflow_integral_local *ij){
    if(kind==2){
        /* The augmented RK is authoritative only for the additive observable.
           Endpoint state/Jacobian authority always comes from the canonical
           terminal flow used by cflow_eval/cflow_eval_jacobian. */
        cflow_local_jac physical;
        if(!cflow_integral_terminal_all_local(x,q,b,h,&physical,ij))return 0;
        cflow_terminal_local(x,q,b,h,1,f);
        return isfinite(f->x)&&isfinite(f->dx)&&isfinite(f->dq)&&isfinite(f->db)
            && isfinite(ij->j)&&isfinite(ij->jq)&&isfinite(ij->jb)&&isfinite(ij->regx);
    }
    int ok=1;
    if(kind==0)cflow_integral_core_local(x,q,b,h,ij);
    else if(kind==1)cflow_integral_highz_local(x,q,b,h,panel,face,ij);
    else if(kind==3)ok=cflow_integral_farfield_local(x,q,b,h,ij);
    else ok=0;
    if(!ok){fallback_quad_n(kind,panel,x,q,b,h,12,ij);}
    cflow_eval_local(kind,panel,face,x,q,b,h,1,f);
    return isfinite(ij->j)&&isfinite(ij->jq)&&isfinite(ij->jb)&&isfinite(f->x);
}

static void all_init(cflow_all_result *o){
    memset(o,0,sizeof(*o));
    o->x=o->dx_dx0=o->dx_dq0=o->dx_db=o->dx_dh=NAN;
    o->integral=o->dI_dx0=o->dI_dq0=o->dI_db=o->dI_dh=o->regularized_dI_dx0=NAN;
    o->event_time=NAN;o->status=CFLOW_OK;
}

static double exact_b0_integral_value(double x,double q,double h){
    if(h==0.0)return 0.0;
    if(q==0.0)return sqrt(x+2.0*h)-sqrt(x);
    double z0=sqrt(x), z1=sqrt(x+2.0*h), dz=z1-z0;
    double theta0=asin(q*x), sum=0.0;
    for(int i=0;i<16;i++){
        double y=cflow_gl16_x[i], z=z0+dz*y;
        double t=0.5*dz*y*(2.0*z0+dz*y);
        double xt=sin(theta0+2.0*q*t)/q;
        if(!(xt>0.0))return NAN;
        sum+=cflow_gl16_w[i]*(z/sqrt(xt));
    }
    return dz*sum;
}


static double event_history_value(double x,double q,double b,double h){
    /* Free-root Gauss rule over the final history segment.  Each node calls the
       endpoint-only public dispatcher, so no integral table is extrapolated and
       no sensitivity stage is evaluated on the branch point itself. */
    if(h==0.0)return 0.0;
    double z0=sqrt(x), z1=sqrt(x+2.0*h), dz=z1-z0, sum=0.0;
    for(int i=0;i<16;i++){
        double y=cflow_gl16_x[i], z=z0+dz*y;
        double t=0.5*dz*y*(2.0*z0+dz*y);
        cflow_eval_result r;cflow_eval(x,q,b,t,&r);
        if(r.status!=CFLOW_OK || !(r.x>0.0) || !isfinite(r.x))return NAN;
        sum+=cflow_gl16_w[i]*(z/sqrt(r.x));
    }
    return dz*sum;
}

static void finish_success(cflow_all_result *o,double x,double q,double x0,double J,double Greg,double Jq,double Jb,double Sx,double Sq,double Sb,size_t steps){
    o->x=x;o->dx_dx0=Sx;o->dx_dq0=Sq;o->dx_db=Sb;
    double zr=q*x,rad=fmax(0.0,1.0-zr*zr);o->dx_dh=2.0*sqrt(rad);
    o->integral=J;o->regularized_dI_dx0=Greg;
    o->dI_dx0=x0>0.0?(-.5/sqrt(x0)+Greg):-INFINITY;
    o->dI_dq0=Jq;o->dI_db=Jb;o->dI_dh=1.0/sqrt(x);o->steps=steps;o->status=CFLOW_OK;
}

static void advance_all_forward(double x0,double q0,double b,double h,cflow_all_result *out){
    all_init(out);out->x=x0;
    if(!isfinite(x0)||!isfinite(q0)||!isfinite(b)||!isfinite(h)||h<0.0){out->status=CFLOW_INVALID_ARGUMENT;return;}
    if(x0<0.0){out->status=CFLOW_OUTSIDE_INTEGRAL_DOMAIN;return;}
    if(!cflow_real_domain(x0,q0)){out->status=CFLOW_OUTSIDE_REAL_DOMAIN;return;}
    if(h==0.0){
        out->x=x0;out->dx_dx0=1.0;out->dx_dq0=out->dx_db=0.0;
        double z=q0*x0;out->dx_dh=2.0*sqrt(fmax(0.0,1.0-z*z));
        out->integral=out->dI_dx0=out->dI_dq0=out->dI_db=out->regularized_dI_dx0=0.0;
        out->dI_dh=x0>0.0?1.0/sqrt(x0):INFINITY;out->steps=0;return;
    }
    if(b==0.0){
        double te=cflow_exact_b0_event(x0,q0);
        if(isfinite(te) && te<=h*(1.0+64.0*DBL_EPSILON)){
            double tol=64.0*DBL_EPSILON*fmax(1.0,fabs(te));
            cflow_local_jac ex;cflow_exact_b0(x0,q0,te,0,&ex);
            double je=exact_b0_integral_value(x0,q0,te);
            if(!isfinite(ex.x)||!isfinite(je)){out->status=CFLOW_NUMERICAL_FAILURE;return;}
            out->x=ex.x;out->integral=je;out->event_time=te;out->steps=1;
            out->status=fabs(h-te)<=tol?CFLOW_EVENT:CFLOW_BEYOND_EVENT;
            out->dx_dx0=out->dx_dq0=out->dx_db=out->dx_dh=NAN;
            out->dI_dx0=out->dI_dq0=out->dI_db=NAN;
            out->regularized_dI_dx0=NAN;
            out->dI_dh=out->status==CFLOW_EVENT?1.0/sqrt(ex.x):NAN;
            return;
        }
    }
    double x=x0,q=q0,t=0.0,rem=h,Sx=1.0,Sq=0.0,Sb=0.0,J=0.0,Jq=0.0,Jb=0.0,Greg=0.0;int first=1;
    int no_terminal_event=cflow_contracting_no_terminal_event(x0,q0,b,h);
    for(size_t step=0;step<CFLOW_MAX_STEPS;step++){
        if(rem==0.0){
            /* The integral cocycle deliberately uses the generic history path
               even at b==0 so J_b is the true first variation toward nonzero b.
               Public endpoint quantities, however, must match the exact-b0
               endpoint API bit-for-bit up to ordinary libm rounding. */
            if(b==0.0){
                cflow_local_jac ex;
                cflow_exact_b0(x0,q0,h,1,&ex);
                if(isfinite(ex.x)){
                    x=ex.x; q=q0;
                    Sx=ex.dx; Sq=ex.dq; Sb=ex.db;
                }
            }
            finish_success(out,x,q,x0,J,Greg,Jq,Jb,Sx,Sq,Sb,step);return;
        }
        if(fabs(q*x)>=1.0){
            out->x=x;out->integral=J;out->event_time=t;out->steps=step;out->status=CFLOW_EVENT;return;
        }
        if(cflow_sep_unresolved(x,q,b,rem)){out->x=x;out->integral=J;out->steps=step;out->status=CFLOW_CONDITIONING_LIMIT;return;}

        cflow_event_local ev;double te=NAN,ex=NAN,eq=NAN,eb=NAN;
        double event_horizon=rem*(1.0+128.0*DBL_EPSILON);
        int has_event=!no_terminal_event && cflow_terminal_event_local_with_horizon(x,q,b,event_horizon,&ev,&te,&ex,&eq,&eb) && te<=event_horizon;

        int kind,panel;cflow_highz_face face;double hs;
        if(!cflow_choose_local(x,q,b,rem,&kind,&panel,&face,&hs)||hs<=0.0){out->x=x;out->integral=J;out->steps=step;out->status=CFLOW_CONDITIONING_LIMIT;return;}
        /* Integral far-field residual has a stricter low-angle domain than the
           endpoint accelerator. Shorten to a legal quotient-core step rather
           than taking an inconsistent endpoint-only history jump. */
        if(kind==3 && fabs(q*x)<sin(0.15)){
            double hc=cflow_core_cap(x,q,b,rem);if(hc>0.0){kind=0;panel=-1;face=CFLOW_HZ_FACE_NONE;hs=hc;}
        }

        if(has_event){
            /* Only the additive history is required on the final event segment:
               endpoint/fixed-time Jacobians are undefined there.  Integrate the
               open interval by the free-root Gaussian fallback, whose nodes are
               strictly interior, then place the endpoint from the event map.
               This avoids forcing an RK sensitivity stage onto the branch point. */
            double je=event_history_value(x,q,b,te);
            if(!isfinite(je)){
                out->x=x;out->integral=J;out->steps=step;out->status=CFLOW_NUMERICAL_FAILURE;return;
            }
            J+=je;
            double total=t+te,qe=q+b*te,sig=(q*x>=0)?1.0:-1.0,xe=sig/qe;
            double tol=128*DBL_EPSILON*fmax(1.0,fabs(total));
            out->x=xe;out->integral=J;out->event_time=total;out->steps=step+1;
            out->status=fabs(h-total)<=tol?CFLOW_EVENT:CFLOW_BEYOND_EVENT;
            out->dx_dx0=out->dx_dq0=out->dx_db=out->dx_dh=NAN;
            out->dI_dx0=out->dI_dq0=out->dI_db=NAN;
            out->regularized_dI_dx0=NAN;
            out->dI_dh=out->status==CFLOW_EVENT?1.0/sqrt(xe):NAN;
            return;
        }

        cflow_integral_local ij;cflow_local_jac f;
        if(!eval_segment_all(kind,panel,face,x,q,b,hs,&f,&ij)){
            out->x=x;out->integral=J;out->steps=step;out->status=CFLOW_NUMERICAL_FAILURE;return;
        }
        J+=ij.j;
        if(first){Greg=ij.regx;Jq+=ij.jq;Jb+=ij.jb;}
        else{Greg+=ij.jx*Sx;Jq+=ij.jx*Sq+ij.jq;Jb+=ij.jx*Sb+ij.jq*t+ij.jb;}
        double nSx=f.dx*Sx,nSq=f.dx*Sq+f.dq,nSb=f.dx*Sb+f.dq*t+f.db;
        Sx=nSx;Sq=nSq;Sb=nSb;x=f.x;q=fma(b,hs,q);t+=hs;
        double old=rem;rem=h-t;if(rem<0&&fabs(rem)<=64*DBL_EPSILON*fmax(h,old))rem=0;if(hs==old)rem=0;first=0;
    }
    out->x=x;out->integral=J;out->steps=CFLOW_MAX_STEPS;out->status=CFLOW_CONDITIONING_LIMIT;
}

void cflow_eval_all(double x0,double q0,double b,double h,cflow_all_result *out){
    if(!out)return;
    advance_all_forward(x0,q0,b,h,out);
}

void cflow_integral_jacobian(double x0,double q0,double b,double h,cflow_integral_jac_result *out){
    if(!out)return;
    cflow_all_result a;advance_all_forward(x0,q0,b,h,&a);
    out->integral=a.integral;out->dI_dx0=a.dI_dx0;out->dI_dq0=a.dI_dq0;out->dI_db=a.dI_db;out->dI_dh=a.dI_dh;
    out->regularized_dI_dx0=a.regularized_dI_dx0;out->event_time=a.event_time;out->steps=a.steps;out->status=a.status;
}



void cflow_integral_value(double x0,double q0,double b,double h,cflow_integral_value_result*out){
    if(!out)return;
    out->integral=NAN;out->event_time=NAN;out->steps=0;out->status=CFLOW_OK;
    if(!isfinite(x0)||!isfinite(q0)||!isfinite(b)||!isfinite(h)||h<0.0){out->status=CFLOW_INVALID_ARGUMENT;return;}if(x0<0.0){out->status=CFLOW_OUTSIDE_INTEGRAL_DOMAIN;return;}if(!cflow_real_domain(x0,q0)){out->status=CFLOW_OUTSIDE_REAL_DOMAIN;return;}if(h==0.0){out->integral=0.0;return;}
    if(b==0.0){double te=cflow_exact_b0_event(x0,q0);if(isfinite(te)&&te<=h*(1.0+64.0*DBL_EPSILON)){double je=exact_b0_integral_value(x0,q0,te);out->integral=je;out->event_time=te;out->steps=1;double tol=64.0*DBL_EPSILON*fmax(1.0,fabs(te));out->status=fabs(h-te)<=tol?CFLOW_EVENT:CFLOW_BEYOND_EVENT;return;}out->integral=exact_b0_integral_value(x0,q0,h);out->steps=1;return;}
    double x=x0,q=q0,t=0.0,rem=h,J=0.0;int no_terminal_event=cflow_contracting_no_terminal_event(x0,q0,b,h);for(size_t step=0;step<CFLOW_MAX_STEPS;step++){
      if(rem==0.0){out->integral=J;out->steps=step;out->status=CFLOW_OK;return;}if(fabs(q*x)>=1.0){out->integral=J;out->event_time=t;out->steps=step;out->status=CFLOW_EVENT;return;}if(cflow_sep_unresolved(x,q,b,rem)){out->integral=J;out->steps=step;out->status=CFLOW_CONDITIONING_LIMIT;return;}
      cflow_event_local ev;double te,ex,eq,eb;double event_horizon=rem*(1.0+128.0*DBL_EPSILON);int has_event=!no_terminal_event&&cflow_terminal_event_local_with_horizon(x,q,b,event_horizon,&ev,&te,&ex,&eq,&eb)&&te<=event_horizon;int kind,panel;cflow_highz_face face;double hs;if(!cflow_choose_local(x,q,b,rem,&kind,&panel,&face,&hs)||hs<=0.0){out->integral=J;out->steps=step;out->status=CFLOW_CONDITIONING_LIMIT;return;}if(kind==3&&fabs(q*x)<sin(0.15)){double hc=cflow_core_cap(x,q,b,rem);if(hc>0.0){kind=0;panel=-1;face=CFLOW_HZ_FACE_NONE;hs=hc;}}
      if(has_event){double je=event_history_value(x,q,b,te);if(!isfinite(je)){out->status=CFLOW_NUMERICAL_FAILURE;return;}J+=je;double total=t+te,tol=128*DBL_EPSILON*fmax(1.0,fabs(total));out->integral=J;out->event_time=total;out->steps=step+1;out->status=fabs(h-total)<=tol?CFLOW_EVENT:CFLOW_BEYOND_EVENT;return;}
      cflow_local_jac f;double j;if(!eval_segment_value(kind,panel,face,x,q,b,hs,&f,&j)){out->integral=J;out->steps=step;out->status=CFLOW_NUMERICAL_FAILURE;return;}J+=j;x=f.x;q=fma(b,hs,q);t+=hs;double old=rem;rem=h-t;if(rem<0&&fabs(rem)<=64*DBL_EPSILON*fmax(h,old))rem=0;if(hs==old)rem=0;
    }out->integral=J;out->steps=CFLOW_MAX_STEPS;out->status=CFLOW_CONDITIONING_LIMIT;
}
