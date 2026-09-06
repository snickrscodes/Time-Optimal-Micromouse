#include "segment_internal.h"

#define C_SMALL_STEP 3.0e-2
#define C_SMALL_TIME 5.0e-2
#define C_SMALL_JAC 5.0e-2
#define C_SMALL_SIGMA 1.7e-1
#define C_ORDER3 3.0e-3
#define C_ORDER4 1.5e-2
#define PI 3.141592653589793238462643383279502884
#define HALF_PI 1.570796326794896619231321691639751442
#define PI_LO 1.2246467991473532e-16
#define J_PI 5.244115108584239
#define J_HALF_PI 2.6220575542921196
#define Q_HALF_PI 1.4441419676288028
#define Q_PI 8.237436749853744

#include "circular_tables.inc"

static double pval(const double *c,int n,double x){double p=c[n-1];for(int i=n-2;i>=0;--i)p=fma(p,x,c[i]);return p;}

static void trig_small(double h,double*out_s,double*out_c,double*out_sinc,double*out_sp,double*out_cm1){
    double y=h*h;
    double st=fma(y,fma(y,fma(y,fma(y,-1.0/39916800.0,1.0/362880.0),-1.0/5040.0),1.0/120.0),-1.0/6.0);
    double sinc=fma(y,st,1.0);
    double ct=fma(y,fma(y,fma(y,fma(y,-1.0/3628800.0,1.0/40320.0),-1.0/720.0),1.0/24.0),-0.5);
    double cm1=y*ct, c=fma(y,ct,1.0);
    double spt=fma(y,fma(y,fma(y,fma(y,1.0/518918400.0,-1.0/3991680.0),1.0/45360.0),-1.0/840.0),1.0/30.0);
    double sp=h*fma(y,spt,-1.0/3.0);
    *out_s=h*sinc;*out_c=c;*out_sinc=sinc;*out_sp=sp;*out_cm1=cm1;
}

static void JQ_left(double x,double *j,double*q){
    if(x==0.0){*j=*q=0.0;return;} double z=x*x,pj=J_COEF[0],pq=Q_COEF[0];
    for(int i=1;i<22;++i){pj=fma(pj,z,J_COEF[i]);pq=fma(pq,z,Q_COEF[i]);}
    double r=sqrt(x);*j=r*pj;*q=x*r*pq;
}
static double J_left(double x){if(x==0.0)return 0.0;double z=x*x,p=J_COEF[0];for(int i=1;i<22;++i)p=fma(p,z,J_COEF[i]);return sqrt(x)*p;}
static void JQ(double x,double*j,double*q){
    if(x==0.0){*j=*q=0;return;} if(x==HALF_PI){*j=J_HALF_PI;*q=Q_HALF_PI;return;} if(x==PI){*j=J_PI;*q=Q_PI;return;}
    if(x<=HALF_PI){JQ_left(x,j,q);return;} double r=(PI-x)+PI_LO,jr,qr;JQ_left(r,&jr,&qr);double rq=fma(-PI,jr,Q_PI);rq=fma(-PI_LO,jr,rq);rq+=qr;*j=J_PI-jr;*q=rq;
}
static double Jonly(double x){if(x==0)return 0;if(x==HALF_PI)return J_HALF_PI;if(x==PI)return J_PI;if(x<=HALF_PI)return J_left(x);return J_PI-J_left((PI-x)+PI_LO);}
static double Htilde(double x,double sx,double j,double q){double y=sqrt(sx);return 2.0*(log1p(y)-atan(y))+fma(-x,j,q);}

static int stable_order(double phase){return phase<C_ORDER3?3:(phase<C_ORDER4?4:5);}
static void stable_series(double x,double z,int count,double*F,double*G,double*TW,double*HS){
    double f[5],tw[5],sg[5]; const double *fa[5]={F0,F1,F2,F3,F4}; int fn[5]={5,11,17,23,29};
    const double *ta[5]={TW0,TW1,TW2,TW3,TW4}; int tn[5]={7,13,19,25,31};
    const double *sa[5]={SG0,SG1,SG2,SG3,SG4}; int sn[5]={6,12,18,24,30};
    for(int i=0;i<count;++i){f[i]=pval(fa[i],fn[i],x);tw[i]=pval(ta[i],tn[i],x);sg[i]=pval(sa[i],sn[i],x);}
    double pf=f[count-1],pg=count*f[count-1],pt=tw[count-1],ps=sg[count-1];
    for(int i=count-2;i>=0;--i){pf=fma(z,pf,f[i]);pg=fma(z,pg,(i+1)*f[i]);pt=fma(z,pt,tw[i]);ps=fma(z,ps,sg[i]);}
    *F=pf;*G=pg;*TW=pt;*HS=ps;
}
static double stable_wsigma(double w0,double k0,double ds,int order){
    double ka=fabs(k0),a=ka*w0*AME_MU_G_INV,d=ka*ds,x,beta,pref;
    const double *p0,*p1,*p2,*p3,*p4; int n0=3,n1=5,n2=7,n3=9,n4=11;
    if(a>=d){x=a==0?0:d/a;beta=a*a;p0=WS_P0_N;p1=WS_P1_N;p2=WS_P2_N;p3=WS_P3_N;p4=WS_P4_N;pref=(k0*AME_MU_G_INV)*ds*ds*w0*w0;}
    else{x=d==0?0:a/d;beta=d*d;p0=WS_P0_NR;p1=WS_P1_NR;p2=WS_P2_NR;p3=WS_P3_NR;p4=WS_P4_NR;double q=AME_MU_G*ds;pref=(k0*AME_MU_G_INV)*ds*ds*q*q;}
    double a0=pval(p0,n0,x),a1=pval(p1,n1,x),a2=pval(p2,n2,x),s;
    if(order==3)s=fma(beta,fma(beta,a2,a1),a0);
    else {double a3=pval(p3,n3,x);if(order==4)s=fma(beta,fma(beta,fma(beta,a3,a2),a1),a0);else{double a4=pval(p4,n4,x);s=fma(beta,fma(beta,fma(beta,fma(beta,a4,a3),a2),a1),a0);}}
    return pref*s;
}

static void local_coeff(double s,double c,ame_circular_ws*w){
    double p=s/c,q=c/s,p2=p*p,p4=p2*p2,p6=p4*p2,p8=p4*p4,p10=p8*p2,p12=p6*p6,p14=p12*p2,p16=p8*p8;
    double q2=q*q,q3=q2*q,q4=q2*q2,q5=q4*q,q6=q3*q3,q7=q6*q,q8=q4*q4,q9=q8*q;
    double t[10]={1.0,-q/4.0,(2*p2+3)*q2/24.0,-(14*p2+15)*q3/192.0,(28*p4+132*p2+105)*q4/1920.0,-(556*p4+1500*p2+945)*q5/23040.0,(1112*p6+10668*p4+19950*p2+10395)*q6/322560.0,-(43784*p6+212940*p4+304290*p2+135135)*q7/5160960.0,(87568*p8+1408992*p6+4533480*p4+5239080*p2+2027025)*q8/92897280.0,-(5723536*p8+43312800*p6+103670280*p4+100540440*p2+34459425)*q9/1857945600.0};
    double tk[10]={0,p/4.0,-1.0/12.0,(14*p2+17)*q/192.0,-(38*p2+39)*q2/480.0,(556*p4+2276*p2+1725)*q3/23040.0,-(2444*p4+6188*p2+3745)*q4/53760.0,(43784*p6+376116*p4+669690*p2+337365)*q5/5160960.0,-(264680*p6+1209996*p4+1662570*p2+717255)*q6/11612160.0,(5723536*p8+84150112*p6+258474600*p4+289101960*p2+109053945)*q7/1857945600.0};
    double ws[15]={0,0,-p/4.0,-(p2+4)/12.0,-(p4+p2+3)*q/24.0,-(3*p4+5*p2-3)/120.0,-(24*p6+50*p4+31*p2-10)*q/1440.0,-(40*p6+98*p4+77*p2+26)/3360.0,-(180*p8+504*p6+483*p4+173*p2+21)*q/20160.0,-(1260*p8+3960*p6+4473*p4+2105*p2+323)/181440.0,-(40320*p10+140400*p8+182448*p6+106300*p4+25261*p2+1284)*q/7257600.0,-(362880*p10+1386000*p8+2035440*p6+1413588*p4+450461*p2+49248)/79833600.0,-(1814400*p12+7539840*p10+12343320*p8+9951216*p6+3984431*p4+675691*p2+24629)*q/479001600.0,-(6652800*p12+29877120*p10+53933880*p8+49500880*p6+23886863*p4+5537805*p2+442249)/2075673600.0,-(479001600*p14+2311545600*p12+4560716160*p10+4694461200*p8+2652321672*p6+783059550*p4+99680491*p2+2653482)*q/174356582400.0};
    double ts[11]={p/6.0,(2*p2-1)/48.0,(4*p4+10*p2+9)*q/240.0,(24*p6+40*p4-84*p2-105)*q2/2880.0,(384*p8+656*p6+796*p4+3284*p2+2775)*q3/80640.0,(1280*p10+2464*p8+1624*p6-6812*p4-21210*p2-13965)*q4/430080.0,(11520*p12+25344*p10+17808*p8+14600*p6+162768*p4+330750*p2+178605)*q5/5806080.0,(161280*p14+403200*p12+334944*p10+118240*p8-840296*p6-4892796*p4-7426440*p2-3399165)*q6/116121600.0,(10321920*p16+29030400*p14+28459008*p12+11270272*p10+7142416*p8+165796032*p6+593976600*p4+719613720*p2+285810525)*q7/10218700800.0,(185794560*pow(p,18)+581898240*p16+661985280*p14+323752704*p12+70636192*p10-764073744*p8-7204438032*p6-18602254440*p4-18752351310*p2-6577696125.0)*q8/245248819200.0,(1857945600*pow(p,20)+6420234240*pow(p,18)+8336148480*p16+4935929856*p14+1274406848*p12+665772640*p10+27349293776*p8+150588364080*p6+303441650820*p4+261903792150*p2+82254647475.0)*q9/3188234649600.0};
    memcpy(w->local_time,t,sizeof(t));memcpy(w->local_tk,tk,sizeof(tk));memcpy(w->local_ws_ext,ws,sizeof(ws));memcpy(w->local_ws,ws,sizeof(w->local_ws));memcpy(w->local_ts_ext,ts,sizeof(ts));memcpy(w->local_ts,ts,sizeof(w->local_ts));
}

ame_segment_status ame_circular_init(ame_segment*s,int stable){
    ame_circular_ws*w=&s->u.circular;if(s->w0<0.0||!ame_finite4(s->L,s->sigma,s->w0,s->k0))return AME_SEGMENT_INVALID_ARGUMENT;
    memset(w,0,sizeof(*w));w->k_abs=fabs(s->k0);w->eps=copysign(1.0,s->k0);w->sqrt_w0=sqrt(s->w0);
    if(s->k0==0.0){w->c0=1.0;w->small_ds_max=w->small_time_ds_max=w->small_jac_ds_max=w->small_sigma_ds_max=INFINITY;w->stable_order=3;return AME_SEGMENT_OK;}
    w->s0=w->k_abs*s->w0*AME_MU_G_INV;if(w->s0>1.0)return AME_SEGMENT_DOMAIN;w->c0=sqrt(fmax(0.0,fma(-w->s0,w->s0,1.0)));w->x0=atan2(w->s0,w->c0);w->domain_end=(HALF_PI-w->x0)/(2*w->k_abs);
    double tol=fmax(128.0*nextafter(fmax(fmax(fabs(w->domain_end),fabs(s->L)),1.0),INFINITY)-128.0*fmax(fmax(fabs(w->domain_end),fabs(s->L)),1.0),1e-14*fmax(1.0,fabs(s->L))); /* overwritten below with ulp helper */
    double mx=fmax(fmax(fabs(w->domain_end),fabs(s->L)),1.0);tol=fmax(128.0*(nextafter(mx,INFINITY)-mx),1e-14*fmax(1.0,fabs(s->L)));
    if(s->L>w->domain_end+tol)return AME_SEGMENT_DOMAIN;
    if(w->s0>0.0&&w->s0<1.0){double rho=2*w->k_abs*fmax(1.0,fmax(w->c0/w->s0,w->s0/w->c0));w->small_ds_max=C_SMALL_STEP/rho;w->small_time_ds_max=C_SMALL_TIME/rho;w->small_jac_ds_max=C_SMALL_JAC/rho;w->small_sigma_ds_max=C_SMALL_SIGMA/rho;w->has_local=1;local_coeff(w->s0,w->c0,w);} 
    if(stable){double phase=w->k_abs*fma(2*AME_MU_G,s->L,s->w0)*AME_MU_G_INV;if(phase>AME_CIRCULAR_STABLE_PHASE_MAX)return AME_SEGMENT_INVALID_ARGUMENT;w->stable_order=stable_order(phase);} 
    if(s->grad&&!stable&&s->k0!=0.0){if(w->c0==0.0)return AME_SEGMENT_INVALID_ARGUMENT;JQ(w->x0,&w->J0,&w->Q0);w->Ht0=Htilde(w->x0,w->s0,w->J0,w->Q0);w->logm0=log1p(-w->s0);w->logp0=log1p(w->s0);w->time_scale=0.5/sqrt(AME_MU_G*w->k_abs);w->sigma_time_scale=0.125*w->eps/(w->k_abs*w->k_abs*sqrt(AME_MU_G*w->k_abs));}
    return AME_SEGMENT_OK;
}
static void stable_state(ame_segment*s,double ds,double*hh,double*sh,double*ch,double*sinc,double*sp,double*w1){ame_circular_ws*w=&s->u.circular;*hh=2*w->k_abs*ds;double cm1;trig_small(*hh,sh,ch,sinc,sp,&cm1);*w1=fma(2*AME_MU_G*w->c0*ds,*sinc,s->w0*(*ch));}

ame_segment_status ame_circular_w(ame_segment*s,double ds,double*out){ame_circular_ws*w=&s->u.circular;if(ds==0){*out=s->w0;return AME_SEGMENT_OK;}if(s->k0==0){*out=fma(2*AME_MU_G,ds,s->w0);return AME_SEGMENT_OK;}if(s->impl==AME_SEGMENT_IMPL_CIRCULAR_STABLE){double h,sh,ch,si,sp;stable_state(s,ds,&h,&sh,&ch,&si,&sp,out);return AME_SEGMENT_OK;}double h=2*w->k_abs*ds;if(w->has_local&&fabs(ds)<=w->small_ds_max){double sh,ch,si,sp,cm1;trig_small(h,&sh,&ch,&si,&sp,&cm1);*out=s->w0*fma(w->c0/w->s0,sh,ch);}else *out=(AME_MU_G/w->k_abs)*sin(w->x0+h);return AME_SEGMENT_OK;}
static void fill_tail(double ds,double*j){j[4]=0;j[5]=ds;j[6]=0;j[7]=1;}
static void local_all(ame_segment*s,double ds,double*outw,double*wj4,double*outt,double*tj4){ame_circular_ws*w=&s->u.circular;double h=2*w->k_abs*ds,sh,ch,si,sp,cm1;trig_small(h,&sh,&ch,&si,&sp,&cm1);double cot=w->c0/w->s0,tan=w->s0/w->c0,g=fma(cot,sh,ch);*outw=s->w0*g;wj4[0]=2*AME_MU_G*fma(w->c0,ch,-w->s0*sh);wj4[2]=fma(-tan,sh,ch);wj4[3]=w->eps*fma(4*AME_MU_G*w->c0*ds*ds,sp,-s->w0*sh*(s->w0/(AME_MU_G*w->c0)+2*ds));wj4[1]=w->eps*s->w0/(w->k_abs*w->k_abs)*pval(w->local_ws,11,h);double sg=sqrt(g),r1=1/(w->sqrt_w0*sg);*outt=(ds/w->sqrt_w0)*pval(w->local_time,10,h);double gm1=fma(cot,sh,cm1),invgm1=-gm1/(sg*(1+sg));tj4[0]=r1;tj4[2]=0.5*invgm1/(AME_MU_G*w->c0*w->sqrt_w0);tj4[3]=(ds/(s->k0*w->sqrt_w0))*pval(w->local_tk,10,h);tj4[1]=w->eps*ds*ds*ds/w->sqrt_w0*pval(w->local_ts,9,h);}
static void local_sigma(ame_segment*s,double ds,double*ws,double*ts){ame_circular_ws*w=&s->u.circular;double h=2*w->k_abs*ds;*ws=w->eps*s->w0/(w->k_abs*w->k_abs)*pval(w->local_ws_ext,15,h);*ts=w->eps*ds*ds*ds/w->sqrt_w0*pval(w->local_ts_ext,11,h);}
static void phase_state(ame_segment*s,double ds,double*h,double*x1,double*sin1,double*cos1){ame_circular_ws*w=&s->u.circular;*h=2*w->k_abs*ds;double sh=sin(*h),ch=cos(*h);*sin1=fma(w->c0,sh,w->s0*ch);*cos1=fma(w->c0,ch,-w->s0*sh);*x1=w->x0+*h;}
static void exact_wjac(ame_segment*s,double ds,double h,double sin1,double cos1,int have_local,double lws,double*outw,double*j){ame_circular_ws*w=&s->u.circular;*outw=(AME_MU_G/w->k_abs)*sin1;j[0]=2*AME_MU_G*cos1;j[2]=cos1/w->c0;j[3]=w->eps*(AME_MU_G/w->k_abs)*fma(cos1,s->w0/(AME_MU_G*w->c0)+2*ds,-sin1/w->k_abs);if(have_local)j[1]=lws;else{double lm=log1p(-sin1),lp=log1p(sin1),lcr=0.5*((lm-w->logm0)+(lp-w->logp0)),c1=fma(-0.5,h*h,lcr);j[1]=-0.5*w->eps*AME_MU_G/pow(w->k_abs,3.0)*fma(cos1,c1,h*sin1);}}
static void exact_tjac(ame_segment*s,double ds,double h,double x1,double sin1,double w1,int have_lj,const double*lj,int have_ls,double lts,double*outt,double*j){ame_circular_ws*w=&s->u.circular;double r1=1/sqrt(w1),j1;if(!have_ls){double q1;JQ(x1,&j1,&q1);double ht1=Htilde(x1,sin1,j1,q1),lm=log1p(-sin1),lp=log1p(sin1),lcr=0.5*((lm-w->logm0)+(lp-w->logp0)),c1=fma(-0.5,h*h,lcr),hd=fma(-h,j1,(w->Ht0-ht1)+(lm-w->logm0));j[1]=w->sigma_time_scale*fma(-2.0/sqrt(sin1),c1,hd);}else{j1=Jonly(x1);j[1]=lts;}*outt=w->time_scale*(j1-w->J0);if(s->w0==0){j[2]=-INFINITY;j[3]=fma(ds,r1,-0.5*(*outt))/s->k0;}else{double r0=1/w->sqrt_w0;j[2]=0.5*(r1-r0)/(AME_MU_G*w->c0);j[3]=fma(s->w0,j[2],fma(ds,r1,-0.5*(*outt)))/s->k0;}j[0]=r1;if(have_lj){j[1]=lj[1];j[2]=lj[2];j[3]=lj[3];}}
static void stable_wjac(ame_segment*s,double ds,double*outw,double*j){ame_circular_ws*w=&s->u.circular;double h,sh,ch,si,sp;stable_state(s,ds,&h,&sh,&ch,&si,&sp,outw);j[0]=2*AME_MU_G*fma(w->c0,ch,-w->s0*sh);j[2]=fma(-w->s0/w->c0,sh,ch);j[3]=w->eps*fma(4*AME_MU_G*w->c0*ds*ds,sp,-s->w0*sh*(s->w0/(AME_MU_G*w->c0)+2*ds));j[1]=stable_wsigma(s->w0,s->k0,ds,stable_order(w->s0+h));}
static void stable_tjac(ame_segment*s,double ds,double w1,double*outt,double*j){ame_circular_ws*w=&s->u.circular;double r2=fma(2*AME_MU_G,ds,s->w0),root=sqrt(r2),dinv=(2*ds)/(root+w->sqrt_w0),d=AME_MU_G*dinv,x=root==0?0:w->sqrt_w0/root,delta=root==0?0:d/root,eta=w->k_abs*r2*AME_MU_G_INV,z=eta*eta,F,G,TW,HS;stable_series(x,z,stable_order(eta),&F,&G,&TW,&HS);*outt=dinv*fma(delta*z,F,1.0);j[0]=1/sqrt(w1);j[3]=(2/s->k0)*dinv*delta*z*G;j[1]=AME_MU_G*dinv*dinv*dinv*z*HS/s->k0;j[2]=w->sqrt_w0==0?-INFINITY:delta*fma(z,TW,-0.5)/(AME_MU_G*w->sqrt_w0);}

ame_segment_status ame_circular_wjac(ame_segment*s,double ds,ame_segment_state_jac*out){if(!s->grad)return AME_SEGMENT_UNSUPPORTED;ame_circular_ws*w=&s->u.circular;if(ds==0){out->w=s->w0;out->jac[0]=2*AME_MU_G*w->c0;out->jac[1]=0;out->jac[2]=1;out->jac[3]=0;fill_tail(ds,out->jac);return AME_SEGMENT_OK;}if(s->k0==0){out->w=fma(2*AME_MU_G,ds,s->w0);out->jac[0]=2*AME_MU_G;out->jac[1]=0;out->jac[2]=1;out->jac[3]=0;fill_tail(ds,out->jac);return AME_SEGMENT_OK;}if(s->impl==AME_SEGMENT_IMPL_CIRCULAR_STABLE){stable_wjac(s,ds,&out->w,out->jac);fill_tail(ds,out->jac);return AME_SEGMENT_OK;}if(w->has_local&&fabs(ds)<=w->small_jac_ds_max){double t,tj[4];local_all(s,ds,&out->w,out->jac,&t,tj);fill_tail(ds,out->jac);return AME_SEGMENT_OK;}double h,x1,si,co,lws=0;phase_state(s,ds,&h,&x1,&si,&co);int hl=w->has_local&&fabs(ds)<=w->small_sigma_ds_max;if(hl){double lts;local_sigma(s,ds,&lws,&lts);}exact_wjac(s,ds,h,si,co,hl,lws,&out->w,out->jac);fill_tail(ds,out->jac);return AME_SEGMENT_OK;}
ame_segment_status ame_circular_tjac(ame_segment*s,double ds,ame_segment_time_jac*out){if(!s->grad)return AME_SEGMENT_UNSUPPORTED;ame_circular_ws*w=&s->u.circular;if(ds==0){out->time=0;out->jac[0]=s->w0==0?INFINITY:1/w->sqrt_w0;out->jac[1]=out->jac[2]=out->jac[3]=0;return AME_SEGMENT_OK;}if(s->k0==0){double r=fma(2*AME_MU_G,ds,s->w0),root=sqrt(r);out->time=(2*ds)/(root+w->sqrt_w0);out->jac[0]=1/root;out->jac[1]=out->jac[3]=0;out->jac[2]=s->w0==0?-INFINITY:0.5*AME_MU_G_INV*(1/root-1/w->sqrt_w0);return AME_SEGMENT_OK;}if(s->impl==AME_SEGMENT_IMPL_CIRCULAR_STABLE){double ww;ame_circular_w(s,ds,&ww);stable_tjac(s,ds,ww,&out->time,out->jac);return AME_SEGMENT_OK;}if(w->has_local&&fabs(ds)<=w->small_ds_max){double ww,wj[4];local_all(s,ds,&ww,wj,&out->time,out->jac);return AME_SEGMENT_OK;}int hlj=0,hls=0;double lj[4]={0},lt=0,lts=0,lws; if(w->has_local&&fabs(ds)<=w->small_jac_ds_max){double ww,wj[4];local_all(s,ds,&ww,wj,&lt,lj);hlj=1;if(fabs(ds)<=w->small_time_ds_max){out->time=lt;memcpy(out->jac,lj,sizeof(lj));return AME_SEGMENT_OK;}}if(w->has_local&&fabs(ds)<=w->small_sigma_ds_max){local_sigma(s,ds,&lws,&lts);hls=1;}double h,x1,si,co;phase_state(s,ds,&h,&x1,&si,&co);double w1=(AME_MU_G/w->k_abs)*si;exact_tjac(s,ds,h,x1,si,w1,hlj,lj,hls,lts,&out->time,out->jac);return AME_SEGMENT_OK;}
ame_segment_status ame_circular_all(ame_segment*s,double ds,ame_segment_all_jac*out){
    if(!s->grad)return AME_SEGMENT_UNSUPPORTED;
    ame_circular_ws*w=&s->u.circular;
    if(ds==0){
        out->w=s->w0;out->state_jac[0]=2*AME_MU_G*w->c0;out->state_jac[1]=0;out->state_jac[2]=1;out->state_jac[3]=0;fill_tail(ds,out->state_jac);
        out->time=0;out->time_jac[0]=s->w0==0?INFINITY:1/w->sqrt_w0;out->time_jac[1]=out->time_jac[2]=out->time_jac[3]=0;return AME_SEGMENT_OK;
    }
    if(s->k0==0){
        out->w=fma(2*AME_MU_G,ds,s->w0);out->state_jac[0]=2*AME_MU_G;out->state_jac[1]=0;out->state_jac[2]=1;out->state_jac[3]=0;fill_tail(ds,out->state_jac);
        double root=sqrt(out->w);out->time=(2*ds)/(root+w->sqrt_w0);out->time_jac[0]=1/root;out->time_jac[1]=out->time_jac[3]=0;out->time_jac[2]=s->w0==0?-INFINITY:0.5*AME_MU_G_INV*(1/root-1/w->sqrt_w0);return AME_SEGMENT_OK;
    }
    if(s->impl==AME_SEGMENT_IMPL_CIRCULAR_STABLE){
        double h,sh,ch,si,sp;
        stable_state(s,ds,&h,&sh,&ch,&si,&sp,&out->w);
        out->state_jac[0]=2*AME_MU_G*fma(w->c0,ch,-w->s0*sh);
        out->state_jac[2]=fma(-w->s0/w->c0,sh,ch);
        out->state_jac[3]=w->eps*fma(4*AME_MU_G*w->c0*ds*ds,sp,-s->w0*sh*(s->w0/(AME_MU_G*w->c0)+2*ds));
        out->state_jac[1]=stable_wsigma(s->w0,s->k0,ds,stable_order(w->s0+h));
        fill_tail(ds,out->state_jac);
        stable_tjac(s,ds,out->w,&out->time,out->time_jac);
        return AME_SEGMENT_OK;
    }
    if(w->has_local&&fabs(ds)<=w->small_time_ds_max){
        double wj4[4];local_all(s,ds,&out->w,wj4,&out->time,out->time_jac);memcpy(out->state_jac,wj4,4*sizeof(double));fill_tail(ds,out->state_jac);return AME_SEGMENT_OK;
    }
    int hlj=0,hls=0;double lj[4]={0},local_time=0,lws=0,lts=0;
    if(w->has_local&&fabs(ds)<=w->small_jac_ds_max){double lw;local_all(s,ds,&lw,lj,&local_time,out->time_jac);hlj=1;}
    if(w->has_local&&fabs(ds)<=w->small_sigma_ds_max){local_sigma(s,ds,&lws,&lts);hls=1;}
    double h,x1,si,co,exact_j[4];phase_state(s,ds,&h,&x1,&si,&co);
    exact_wjac(s,ds,h,si,co,hls,lws,&out->w,exact_j);
    if(hlj)memcpy(out->state_jac,lj,4*sizeof(double));else memcpy(out->state_jac,exact_j,4*sizeof(double));
    fill_tail(ds,out->state_jac);
    exact_tjac(s,ds,h,x1,si,out->w,hlj,lj,hls,lts,&out->time,out->time_jac);
    return AME_SEGMENT_OK;
}
ame_segment_status ame_circular_time(ame_segment*s,double ds,double*out){if(!s->grad)return AME_SEGMENT_UNSUPPORTED;ame_segment_time_jac t;ame_segment_status st=ame_circular_tjac(s,ds,&t);*out=t.time;return st;}
