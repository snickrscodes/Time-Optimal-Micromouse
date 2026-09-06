#include "ame_dd_yaw.h"
#include <math.h>
#include <stddef.h>
#include <string.h>

static int finite7(double a,double b,double c,double d,double e,double f,double g){return isfinite(a)&&isfinite(b)&&isfinite(c)&&isfinite(d)&&isfinite(e)&&isfinite(f)&&isfinite(g);}
static int side_eval(const ame_dd_params*p,int eps,double w,double k,double sigma,double *lo,double *hi,ame_dd_candidate*out){
    if(!p||!isfinite(w)||!isfinite(k)||!isfinite(sigma)||w<0)return AME_DD_INVALID;
    if(out && w<=0)return AME_DD_INVALID;
    double r=sqrt(w),h=1.0+eps*p->beta*k,c=1.0+eps*p->eta*k;
    if(!(h>p->h_floor&&c>p->c_floor))return AME_DD_DOMAIN;
    double vs=r*h;
    if(!(vs<p->v_free-p->speed_margin))return AME_DD_DOMAIN;
    double Q=p->q0*(1.0-vs/p->v_free);
    if(!(Q>0&&isfinite(Q)))return AME_DD_DOMAIN;
    double d=eps*p->eta*w*sigma;
    double L=(-Q-d)/c,U=(Q-d)/c;
    if(lo) *lo=L;
    if(hi) *hi=U;
    if(out){
        double Qw=-p->q0*h/(2.0*p->v_free*r);
        double Qk=-p->q0*r*eps*p->beta/p->v_free;
        out->value=U;
        out->dw=(Qw-eps*p->eta*sigma)/c;
        out->dk=(Qk-eps*p->eta*U)/c;
        out->dsigma=-eps*p->eta*w/c;
    }
    return AME_DD_OK;
}
int ame_dd_candidate_eval(const ame_dd_params*p,const ame_dd_profile*q,int mode,int kind,double w,double k,double sigma,ame_dd_candidate*out){
    if(!p||!q||!out||!finite7(p->beta,p->eta,p->q0,p->v_free,q->a_max,q->b_emf,q->mu_g)||w<=0||!isfinite(k)||!isfinite(sigma))return AME_DD_INVALID;
    memset(out,0,sizeof(*out));
    if(mode==AME_DD_MOTOR){if(kind!=AME_DD_FORWARD)return AME_DD_INVALID;double r=sqrt(w);out->value=q->a_max-q->b_emf*r;out->dw=-q->b_emf/(2*r);return AME_DD_OK;}
    if(mode==AME_DD_BRAKE){if(kind!=AME_DD_BACKWARD)return AME_DD_INVALID;out->value=q->a_brake;return AME_DD_OK;}
    if(mode==AME_DD_GRIP){double z=w*k,g2=q->mu_g*q->mu_g-z*z;if(!(g2>0))return AME_DD_DOMAIN;double g=sqrt(g2);out->value=g;out->dw=-z*k/g;out->dk=-z*w/g;return AME_DD_OK;}
    if(mode==AME_DD_SIDE_RIGHT)return side_eval(p,1,w,k,sigma,NULL,NULL,out);
    if(mode==AME_DD_SIDE_LEFT)return side_eval(p,-1,w,k,sigma,NULL,NULL,out);
    return AME_DD_INVALID;
}
int ame_dd_interval_eval(const ame_dd_params*p,const ame_dd_profile*q,double w,double k,double sigma,ame_dd_interval*out){
    if(!p||!q||!out||w<0||!isfinite(w)||!isfinite(k)||!isfinite(sigma))return AME_DD_INVALID;
    double r=sqrt(w),z=w*k,g2=q->mu_g*q->mu_g-z*z;if(!(g2>=0))return AME_DD_DOMAIN;double g=sqrt(g2);
    double ll,lu,rl,ru;int st=side_eval(p,-1,w,k,sigma,&ll,&lu,NULL);if(st)return st;st=side_eval(p,1,w,k,sigma,&rl,&ru,NULL);if(st)return st;
    double motor=q->a_max-q->b_emf*r;
    double lo=-q->a_brake;if(-g>lo)lo=-g;if(ll>lo)lo=ll;if(rl>lo)lo=rl;
    double hi=motor;if(g<hi)hi=g;if(lu<hi)hi=lu;if(ru<hi)hi=ru;
    out->lower=lo;out->upper=hi;out->margin=hi-lo;out->motor_upper=motor;out->grip=g;out->left_lower=ll;out->left_upper=lu;out->right_lower=rl;out->right_upper=ru;return AME_DD_OK;
}

static int rhs(const ame_dd_params*p,const ame_dd_profile*q,int mode,int kind,double sigma,double k0,double s,const double y[8],double z[8]){
    double w=y[0];if(!(w>0&&isfinite(w)))return AME_DD_DOMAIN;double k=fma(sigma,s,k0);ame_dd_candidate c;int st=ame_dd_candidate_eval(p,q,mode,kind,w,k,sigma,&c);if(st)return st;
    double f=2*c.value,fw=2*c.dw,fk=2*c.dk,fs=2*c.dsigma;
    double ss=y[2],sw=y[3],sk=y[4],rt=sqrt(w),qq=-0.5/(w*rt);
    z[0]=f;z[1]=1/rt;z[2]=fw*ss+fk*s+fs;z[3]=fw*sw;z[4]=fw*sk+fk;z[5]=qq*ss;z[6]=qq*sw;z[7]=qq*sk;return AME_DD_OK;
}
static double errnorm(const double y[8],const double y5[8],const double e[8]){
    static const double at[8]={2e-12,2e-12,2e-11,2e-11,2e-11,2e-11,2e-11,2e-11};double m=0;
    for(int i=0;i<8;i++){double sc=at[i]+3e-11*fmax(fabs(y[i]),fabs(y5[i]));double v=fabs(e[i])/sc;if(v>m)m=v;}return m;
}
int ame_dd_segment_eval(const ame_dd_params*p,const ame_dd_profile*q,int mode,int kind,double L,double sigma,double w0,double k0,double ds,ame_dd_segment_all*out){
    if(!p||!q||!out||!isfinite(L)||!isfinite(ds)||!isfinite(sigma)||!isfinite(w0)||!isfinite(k0)||L<0||ds<0||ds>L+1e-12||w0<=0)return AME_DD_INVALID;
    if(ds>L) ds=L;
    double y[8]={w0,0,0,1,0,0,0,0}; double x=0;
    if(ds==0){
        ame_dd_candidate c; int st=ame_dd_candidate_eval(p,q,mode,kind,w0,k0,sigma,&c); if(st)return st;
        out->w=w0; double sj[8]={2*c.value,0,1,0,sigma,0,0,1}; memcpy(out->state_jac,sj,sizeof(sj));
        out->time=0; double tj[4]={1/sqrt(w0),0,0,0}; memcpy(out->time_jac,tj,sizeof(tj)); return AME_DD_OK;
    }
    double h=fmin(0.02,fmax(ds/16.0,1e-5));int steps=0;
    while(x<ds){if(++steps>200000)return AME_DD_NUMERICAL;if(h>ds-x)h=ds-x;
        double k1[8],k2[8],k3[8],k4[8],k5[8],k6[8],k7[8],yt[8],y5[8],y4[8],er[8];int st;
#define EVAL(K,C2,A1,K1,A2,K2,A3,K3,A4,K4,A5,K5,A6,K6) do{for(int i=0;i<8;i++)yt[i]=y[i]+h*((A1)*K1[i]+(A2)*K2[i]+(A3)*K3[i]+(A4)*K4[i]+(A5)*K5[i]+(A6)*K6[i]);st=rhs(p,q,mode,kind,sigma,k0,x+(C2)*h,yt,K);if(st)return st;}while(0)
        st=rhs(p,q,mode,kind,sigma,k0,x,y,k1);if(st)return st;
        EVAL(k2,1.0/5,1.0/5,k1,0,k1,0,k1,0,k1,0,k1,0,k1);
        EVAL(k3,3.0/10,3.0/40,k1,9.0/40,k2,0,k1,0,k1,0,k1,0,k1);
        EVAL(k4,4.0/5,44.0/45,k1,-56.0/15,k2,32.0/9,k3,0,k1,0,k1,0,k1);
        EVAL(k5,8.0/9,19372.0/6561,k1,-25360.0/2187,k2,64448.0/6561,k3,-212.0/729,k4,0,k1,0,k1);
        EVAL(k6,1.0,9017.0/3168,k1,-355.0/33,k2,46732.0/5247,k3,49.0/176,k4,-5103.0/18656,k5,0,k1);
        EVAL(k7,1.0,35.0/384,k1,0,k2,500.0/1113,k3,125.0/192,k4,-2187.0/6784,k5,11.0/84,k6);
        for(int i=0;i<8;i++){
            y5[i]=y[i]+h*(35.0/384*k1[i]+500.0/1113*k3[i]+125.0/192*k4[i]-2187.0/6784*k5[i]+11.0/84*k6[i]);
            y4[i]=y[i]+h*(5179.0/57600*k1[i]+7571.0/16695*k3[i]+393.0/640*k4[i]-92097.0/339200*k5[i]+187.0/2100*k6[i]+1.0/40*k7[i]);er[i]=y5[i]-y4[i];
        }
        double en=errnorm(y,y5,er);
        if(en<=1.0){memcpy(y,y5,sizeof(y));x+=h;double fac=en==0?5.0:0.9*pow(en,-0.2);if(fac<0.2)fac=0.2;if(fac>5)fac=5;h*=fac;}
        else{double fac=0.9*pow(en,-0.25);if(fac<0.1)fac=0.1;if(fac>0.5)fac=0.5;h*=fac;if(h<1e-14)return AME_DD_NUMERICAL;}
#undef EVAL
    }
    if(!(y[0]>0&&isfinite(y[0]))) return AME_DD_DOMAIN;
    ame_dd_candidate c; int st=ame_dd_candidate_eval(p,q,mode,kind,y[0],fma(sigma,ds,k0),sigma,&c); if(st)return st;
    out->w=y[0];out->state_jac[0]=2*c.value;out->state_jac[1]=y[2];out->state_jac[2]=y[3];out->state_jac[3]=y[4];out->state_jac[4]=sigma;out->state_jac[5]=ds;out->state_jac[6]=0;out->state_jac[7]=1;out->time=y[1];out->time_jac[0]=1/sqrt(y[0]);out->time_jac[1]=y[5];out->time_jac[2]=y[6];out->time_jac[3]=y[7];return AME_DD_OK;
}
const char *ame_dd_status_name(int s){switch(s){case AME_DD_OK:return"ok";case AME_DD_INVALID:return"invalid";case AME_DD_DOMAIN:return"domain";case AME_DD_NUMERICAL:return"numerical";default:return"unknown";}}

static double mvc_hard_cap_w(const ame_dd_params *p,const ame_dd_profile *q,double k){
    double cap=INFINITY,ak=fabs(k);
    if(ak>0.0) cap=q->mu_g/ak;
    for(int ei=0;ei<2;ei++){
        int eps=ei==0?1:-1;
        double h=1.0+eps*p->beta*k,c=1.0+eps*p->eta*k;
        if(!(h>p->h_floor&&c>p->c_floor)) return 0.0;
        double wc=(p->v_free/h)*(p->v_free/h);
        if(wc<cap) cap=wc;
    }
    if(!isfinite(cap)) cap=p->v_free*p->v_free;
    return cap>0.0?cap:0.0;
}
static double mvc_margin(const ame_dd_params*p,const ame_dd_profile*q,double w,double k,double sigma){
    ame_dd_interval iv; int st=ame_dd_interval_eval(p,q,w,k,sigma,&iv);
    return st==AME_DD_OK?iv.margin:-INFINITY;
}
static void mvc_labels(const ame_dd_params*p,const ame_dd_profile*q,double w,double k,double sigma,int *um,int *lm,double *margin){
    ame_dd_interval iv; int st=ame_dd_interval_eval(p,q,w,k,sigma,&iv);
    if(st!=AME_DD_OK){*um=0;*lm=0;if(margin)*margin=-INFINITY;return;}
    double uv[4]={iv.motor_upper,iv.grip,iv.right_upper,iv.left_upper};
    int ul[4]={AME_DD_MOTOR,AME_DD_GRIP,AME_DD_SIDE_RIGHT,AME_DD_SIDE_LEFT};
    int ui=0;for(int i=1;i<4;i++)if(uv[i]<uv[ui])ui=i;
    double lv[4]={-q->a_brake,-iv.grip,iv.right_lower,iv.left_lower};
    int ll[4]={AME_DD_BRAKE,AME_DD_GRIP,AME_DD_SIDE_RIGHT,AME_DD_SIDE_LEFT};
    int li=0;for(int i=1;i<4;i++)if(lv[i]>lv[li])li=i;
    *um=ul[ui];*lm=ll[li];if(margin)*margin=iv.margin;
}
static int mvc_scan_one(const ame_dd_params*p,const ame_dd_profile*q,double k,double sigma,int n_scan,ame_dd_mvc_point*out){
    if(!p||!q||!out||!isfinite(k)||!isfinite(sigma))return AME_DD_INVALID;
    memset(out,0,sizeof(*out));
    double hard=mvc_hard_cap_w(p,q,k);out->hard_cap_w=hard;
    if(!(hard>0.0)){out->w=0.0;out->margin=-INFINITY;out->status=AME_DD_DOMAIN;return AME_DD_OK;}
    double hi=nextafter(hard,0.0);if(!(hi>0.0))hi=0.5*hard;
    double lo=1e-12*fmax(1.0,hi);if(lo<1e-12)lo=1e-12;
    double h0=mvc_margin(p,q,lo,k,sigma);
    if(!isfinite(h0)||h0<0.0){out->w=lo;out->margin=h0;out->status=AME_DD_DOMAIN;return AME_DD_OK;}
    int ns=n_scan<24?24:n_scan;
    double y0=sqrt(lo),y1=sqrt(hi),prev_w=lo,prev_h=h0,a=0.0,b=0.0;int found=0;
    for(int i=1;i<=ns;i++){
        double y=y0+(y1-y0)*(double)i/(double)ns,w=y*y,h=mvc_margin(p,q,w,k,sigma);
        if(prev_h>=0.0&&(!isfinite(h)||h<=0.0)){a=prev_w;b=w;found=1;break;}
        prev_w=w;prev_h=h;
    }
    double cap=hi;
    if(found){
        /* Monotone bracket refinement of the first connected feasible boundary.
           The reference uses Brent on the same bracket.  Bisection is deliberately
           conservative and stops well below the public MVC tolerances. */
        for(int it=0;it<96;it++){
            double m=0.5*(a+b),hm=mvc_margin(p,q,m,k,sigma);
            if(isfinite(hm)&&hm>0.0)a=m;else b=m;
            double tol=fmax(1e-13,4e-13*fmax(1.0,fabs(m)));
            if(b-a<=tol)break;
        }
        cap=0.5*(a+b);
    }
    double wc=fmax(lo,nextafter(cap,0.0)),im=0.0;int um=0,lm=0;
    mvc_labels(p,q,wc,k,sigma,&um,&lm,&im);
    double cm=mvc_margin(p,q,cap,k,sigma);if(!isfinite(cm))cm=im;
    out->w=cap;out->margin=cm;out->upper_mode=um;out->lower_mode=lm;out->status=AME_DD_OK;
    return AME_DD_OK;
}
int ame_dd_mvc_scan_bulk(const ame_dd_params*p,const ame_dd_profile*q,const double*kappa,const double*sigma,size_t count,int n_scan,ame_dd_mvc_point*out){
    if(!p||!q||(!kappa&&count)||(!sigma&&count)||(!out&&count))return AME_DD_INVALID;
    for(size_t i=0;i<count;i++){
        int st=mvc_scan_one(p,q,kappa[i],sigma[i],n_scan,&out[i]);
        if(st!=AME_DD_OK)return st;
    }
    return AME_DD_OK;
}
