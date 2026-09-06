#include "include/ame_crossing.h"
#include "../segment/include/ame_segment_constants.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cfloat>
#include <limits>
#include <vector>

namespace {

constexpr double M = AME_SEGMENT_MU_G;
constexpr double A_BRAKE = AME_SEGMENT_A_BRAKE;
constexpr double A_MAX = AME_SEGMENT_A_MAX;
constexpr double B_EMF = AME_SEGMENT_B_EMF;
constexpr double M2 = M * M;
constexpr double S_BRAKE = 2.0 * A_BRAKE;
constexpr int CERT_ULPS = 16;
constexpr double POS_TO_NEG = 1.0;
constexpr double NEG_TO_POS = -1.0;

inline bool isfin(double x) { return std::isfinite(x); }
inline double qnan() { return std::numeric_limits<double>::quiet_NaN(); }
inline double pinf() { return std::numeric_limits<double>::infinity(); }
inline double ninf() { return -std::numeric_limits<double>::infinity(); }

static double shift_ulps(double value, double toward, int count = CERT_ULPS) {
    double out = value;
    if (!isfin(out)) return out;
    for (int i = 0; i < count; ++i) out = std::nextafter(out, toward);
    return out;
}

static const double K_BRAKE = std::sqrt(std::fma(-A_BRAKE, A_BRAKE, M2));
static const double K_BRAKE_LO = shift_ulps(K_BRAKE, ninf());
static const double K_BRAKE_HI = shift_ulps(K_BRAKE, pinf());

struct Eval {
    double f = qnan();
    double df = qnan();
    bool has_df = false;
    double g2 = qnan();
    bool outside = false;
    double native_event = qnan();
};

enum class EvalKind { MotorGrip, GripMotor, GripBrake, BrakeGrip };

struct Ctx {
    ame_segment *seg = nullptr;
    double L = 0.0;
    double sigma = 0.0;
    double w0 = 0.0;
    double k0 = 0.0;
    ame_segment_impl impl = static_cast<ame_segment_impl>(0);
    uint64_t evals = 0;
};

static void clear_result(ame_crossing_result *r) {
    *r = {};
    r->event = qnan();
    r->domain_edge = qnan();
    r->domain_safe = qnan();
    r->domain_event = qnan();
}

static void set_bracket(ame_crossing_bracket &b, double lo, double hi, double flo, double fhi) {
    b.present = 1; b.lo = lo; b.hi = hi; b.f_lo = flo; b.f_hi = fhi;
}

static ame_crossing_status crossing_state(Ctx &ctx, double ds, ame_segment_crossing_state &st) {
    ++ctx.evals;
    ame_segment_status s = ame_segment_crossing_state_at(ctx.seg, ds, &st);
    if (s == AME_SEGMENT_OK) return AME_CROSSING_OK;
    if (s == AME_SEGMENT_INVALID_ARGUMENT || s == AME_SEGMENT_OUTSIDE_PREFIX) return AME_CROSSING_INVALID_ARGUMENT;
    return AME_CROSSING_SEGMENT_ERROR;
}

static ame_crossing_status eval_at(Ctx &ctx, EvalKind kind, double ds, Eval &out) {
    ame_segment_crossing_state st{};
    ame_crossing_status cs = crossing_state(ctx, ds, st);
    if (cs != AME_CROSSING_OK) return cs;
    out.g2 = st.g2;
    out.outside = st.outside_domain != 0;
    out.native_event = out.outside ? st.event_position : qnan();
    if (out.outside) {
        out.f = qnan(); out.has_df = false; out.df = qnan(); out.g2 = ninf();
        return AME_CROSSING_OK;
    }
    const double w = st.w, k = st.k, q = st.q, g2 = st.g2;
    if (!isfin(w) || w < 0.0) {
        out.f = qnan(); out.has_df = false; return AME_CROSSING_OK;
    }
    if (kind == EvalKind::MotorGrip || kind == EvalKind::GripMotor) {
        const double y = std::sqrt(w);
        if (!(g2 > 0.0) || !isfin(g2)) {
            out.f = std::fma(B_EMF, y, -A_MAX); out.has_df = false; return AME_CROSSING_OK;
        }
        const double g = std::sqrt(g2);
        out.f = std::fma(B_EMF, y, g - A_MAX);
        double dw;
        if (kind == EvalKind::MotorGrip) dw = 2.0 * std::fma(B_EMF, -y, A_MAX);
        else dw = 2.0 * g;
        const double qp = std::fma(k, dw, ctx.sigma * w);
        if (y == 0.0) out.has_df = false;
        else { out.df = std::fma(-q / g, qp, 0.5 * B_EMF * dw / y); out.has_df = isfin(out.df); }
        return AME_CROSSING_OK;
    }
    if (kind == EvalKind::GripBrake || kind == EvalKind::BrakeGrip) {
        if (!(g2 > 0.0) || !isfin(g2)) {
            out.f = -A_BRAKE; out.has_df = false; return AME_CROSSING_OK;
        }
        const double g = std::sqrt(g2);
        out.f = g - A_BRAKE;
        const double dw = kind == EvalKind::GripBrake ? 2.0 * g : S_BRAKE;
        const double qp = std::fma(k, dw, ctx.sigma * w);
        out.df = -q * qp / g;
        out.has_df = isfin(out.df);
        return AME_CROSSING_OK;
    }
    return AME_CROSSING_INVALID_ARGUMENT;
}

static bool crosses(double fa, double fb, double dir) {
    if (!isfin(fa) || !isfin(fb)) return false;
    if (dir == POS_TO_NEG) return fa >= 0.0 && fb <= 0.0;
    return fa <= 0.0 && fb >= 0.0;
}

static bool already_target(double f, double dir) {
    return dir == POS_TO_NEG ? f <= 0.0 : f >= 0.0;
}

static bool initial_switch_required(double f, const Eval &e, double dir, double residual_tol, double spatial_tol) {
    if (!isfin(f)) return false;
    bool near = std::fabs(f) <= residual_tol;
    if (e.has_df && e.df != 0.0) near = near || std::fabs(f / e.df) <= spatial_tol;
    if (near) {
        if (!e.has_df) return false;
        return dir == POS_TO_NEG ? e.df < 0.0 : e.df > 0.0;
    }
    return already_target(f, dir);
}

static double locator_spatial_tol(const ame_crossing_options &o, double L) {
    double v = o.x_abs_tol + o.x_rel_tol * std::max(1.0, std::fabs(L));
    if (o.has_initial_spatial_tol) v = std::max(v, o.initial_spatial_tol);
    return v;
}

static ame_crossing_status root_solve(Ctx &ctx, EvalKind kind, double lo, double hi, double flo, double fhi,
                                      const ame_crossing_options &o, double &root) {
    if (hi < lo) { std::swap(lo, hi); std::swap(flo, fhi); }
    if (flo == 0.0) { root = lo; return AME_CROSSING_OK; }
    if (fhi == 0.0) { root = hi; return AME_CROSSING_OK; }
    if (!isfin(flo) || !isfin(fhi) || flo * fhi > 0.0) return AME_CROSSING_NUMERICAL_FAILURE;
    double x;
    if (fhi != flo) {
        x = hi - fhi * (hi - lo) / (fhi - flo);
        if (!(lo < x && x < hi) || !isfin(x)) x = 0.5 * (lo + hi);
    } else x = 0.5 * (lo + hi);
    for (int it = 0; it < o.max_iter; ++it) {
        Eval ex{}; ame_crossing_status st = eval_at(ctx, kind, x, ex); if (st != AME_CROSSING_OK) return st;
        double fx = ex.outside ? qnan() : ex.f;
        if (!isfin(fx)) {
            x = 0.5 * (lo + hi); st = eval_at(ctx, kind, x, ex); if (st != AME_CROSSING_OK) return st;
            fx = ex.outside ? qnan() : ex.f; if (!isfin(fx)) return AME_CROSSING_NUMERICAL_FAILURE;
        }
        if (std::fabs(fx) <= o.f_tol) { root = x; return AME_CROSSING_OK; }
        if (flo * fx <= 0.0) { hi = x; fhi = fx; }
        else { lo = x; flo = fx; }
        const double mid = 0.5 * (lo + hi);
        if (hi - lo <= o.x_abs_tol + o.x_rel_tol * std::max(1.0, std::fabs(mid))) { root = mid; return AME_CROSSING_OK; }
        if (ex.has_df && ex.df != 0.0 && isfin(ex.df)) {
            const double xn = x - fx / ex.df;
            if (lo < xn && xn < hi && isfin(xn)) { x = xn; continue; }
        }
        if (fhi != flo) {
            const double xs = hi - fhi * (hi - lo) / (fhi - flo);
            if (lo < xs && xs < hi && isfin(xs)) { x = xs; continue; }
        }
        x = mid;
    }
    root = 0.5 * (lo + hi);
    return AME_CROSSING_OK;
}

static ame_crossing_status bisect_domain(Ctx &ctx, EvalKind kind, double lo, double hi, double g_lo, double g_hi,
                                         double margin, double &edge) {
    if (g_lo <= margin) { edge = lo; return AME_CROSSING_OK; }
    if (g_hi > margin) return AME_CROSSING_NUMERICAL_FAILURE;
    double a = lo, b = hi;
    for (int i = 0; i < 80; ++i) {
        const double mid = 0.5 * (a + b);
        Eval e{}; ame_crossing_status st = eval_at(ctx, kind, mid, e);
        double gm = ninf();
        if (st == AME_CROSSING_OK && !e.outside) gm = e.g2;
        else if (st != AME_CROSSING_OK && st != AME_CROSSING_SEGMENT_ERROR) return st;
        if (isfin(gm) && gm > margin) a = mid; else b = mid;
    }
    edge = b; return AME_CROSSING_OK;
}

static ame_crossing_status generic_scan(Ctx &ctx, EvalKind kind, double dir, const ame_crossing_options &o,
                                        ame_crossing_result &r) {
    Eval prev{}; ame_crossing_status st = eval_at(ctx, kind, 0.0, prev);
    if (st != AME_CROSSING_OK) {
        r.has_domain_edge = 1; r.domain_edge = 0.0; set_bracket(r.domain_bracket,0,0,qnan(),ninf()); return AME_CROSSING_OK;
    }
    if (!isfin(prev.g2) || (prev.g2 <= o.domain_margin && !o.allow_initial_boundary)) {
        r.has_domain_edge=1; r.domain_edge=0.0; set_bracket(r.domain_bracket,0,0,prev.g2,prev.g2); return AME_CROSSING_OK;
    }
    const double initial_tol = locator_spatial_tol(o, ctx.L);
    if (initial_switch_required(prev.f, prev, dir, o.f_tol, initial_tol)) {
        r.has_event=1; r.event=0.0; r.initial_switch=1; set_bracket(r.event_bracket,0,0,prev.f,prev.f); return AME_CROSSING_OK;
    }
    if (ctx.L == 0.0) return AME_CROSSING_OK;
    double s_prev = 0.0;
    for (int j = 1; j <= o.n_scan; ++j) {
        const double s = ctx.L * (static_cast<double>(j) / o.n_scan);
        Eval e{}; st = eval_at(ctx, kind, s, e);
        if (st != AME_CROSSING_OK) { e.f=qnan(); e.g2=ninf(); e.outside=true; }
        if (!isfin(e.g2) || e.g2 <= o.domain_margin || e.outside) {
            double edge;
            st = bisect_domain(ctx, kind, s_prev, s, prev.g2, isfin(e.g2)?e.g2:ninf(), o.domain_margin, edge);
            if (st != AME_CROSSING_OK) return st;
            Eval ee{}; st = eval_at(ctx, kind, edge, ee); if (st != AME_CROSSING_OK && st != AME_CROSSING_SEGMENT_ERROR) return st;
            const double fedge = (!ee.outside && st==AME_CROSSING_OK)?ee.f:qnan();
            set_bracket(r.domain_bracket,s_prev,s,prev.g2,e.g2);
            r.has_domain_edge=1; r.domain_edge=edge;
            if (crosses(prev.f, fedge, dir)) {
                set_bracket(r.event_bracket,s_prev,edge,prev.f,fedge);
                double root; st = root_solve(ctx,kind,s_prev,edge,prev.f,fedge,o,root);
                if (st==AME_CROSSING_OK && root<=edge) {r.has_event=1;r.event=root;}
            }
            return AME_CROSSING_OK;
        }
        if (crosses(prev.f,e.f,dir)) {
            set_bracket(r.event_bracket,s_prev,s,prev.f,e.f);
            double root; st=root_solve(ctx,kind,s_prev,s,prev.f,e.f,o,root); if(st!=AME_CROSSING_OK)return st;
            r.has_event=1;r.event=root; return AME_CROSSING_OK;
        }
        s_prev=s; prev=e;
    }
    return AME_CROSSING_OK;
}

struct DomainRefined { double safe, edge, gsafe, gedge; };
static ame_crossing_status refine_grip_domain(Ctx &ctx, EvalKind kind, double lo, double hi, double g_lo,
                                              const ame_crossing_options &o, DomainRefined &d) {
    if (!(g_lo > o.physical_domain_margin)) { d={lo,lo,g_lo,g_lo}; return AME_CROSSING_OK; }
    double a=lo,b=hi,ga=g_lo,gb=ninf();
    for(int i=0;i<80;++i){
        double mid=0.5*(a+b); if(mid==a||mid==b)break;
        Eval e{}; ame_crossing_status st=eval_at(ctx,kind,mid,e);
        if(st!=AME_CROSSING_OK)return st;
        double gm=(!e.outside)?e.g2:ninf();
        if(!e.outside&&isfin(gm)&&gm>o.physical_domain_margin){a=mid;ga=gm;}else{b=mid;gb=gm;}
    }
    double safe=a,gsafe=ga;
    if(g_lo>=o.domain_stop_margin && ga<o.domain_stop_margin){
        double c=lo,dd=a,gc=g_lo;
        for(int i=0;i<80;++i){double mid=0.5*(c+dd);if(mid==c||mid==dd)break;Eval e{};auto st=eval_at(ctx,kind,mid,e);if(st!=AME_CROSSING_OK)return st;double gm=(!e.outside)?e.g2:ninf();if(!e.outside&&isfin(gm)&&gm>=o.domain_stop_margin){c=mid;gc=gm;}else dd=mid;}safe=c;gsafe=gc;
    }
    d={safe,b,gsafe,gb};return AME_CROSSING_OK;
}

static ame_crossing_status safe_grip_scan(Ctx &ctx, EvalKind kind, double dir, const ame_crossing_options &o,
                                          ame_crossing_result &r) {
    Eval prev{}; ame_crossing_status st=eval_at(ctx,kind,0.0,prev);
    if(st!=AME_CROSSING_OK || prev.outside || !isfin(prev.g2)){
        r.has_domain_edge=1;r.domain_edge=0.0;r.has_domain_safe=1;r.domain_safe=0.0;
        if(prev.outside&&isfin(prev.native_event)){r.has_domain_event=1;r.domain_event=prev.native_event;}
        set_bracket(r.domain_bracket,0,0,prev.g2,prev.g2);return AME_CROSSING_OK;
    }
    if(prev.g2<=o.physical_domain_margin&&!o.allow_initial_boundary){r.has_domain_edge=1;r.domain_edge=0.0;r.has_domain_safe=1;r.domain_safe=0;set_bracket(r.domain_bracket,0,0,prev.g2,prev.g2);return AME_CROSSING_OK;}
    const double initial_tol=locator_spatial_tol(o,ctx.L);
    if(initial_switch_required(prev.f,prev,dir,o.f_tol,initial_tol)){r.has_event=1;r.event=0;r.initial_switch=1;set_bracket(r.event_bracket,0,0,prev.f,prev.f);return AME_CROSSING_OK;}
    if(ctx.L==0)return AME_CROSSING_OK;
    double s_prev=0;
    for(int j=1;j<=o.n_scan;++j){
        double s=ctx.L*(static_cast<double>(j)/o.n_scan);Eval e{};st=eval_at(ctx,kind,s,e);
        if(st!=AME_CROSSING_OK)return st;
        bool outside=e.outside;
        bool physical=outside||!isfin(e.g2)||e.g2<=o.physical_domain_margin;
        if(physical){
            DomainRefined d{};st=refine_grip_domain(ctx,kind,s_prev,s,prev.g2,o,d);if(st!=AME_CROSSING_OK)return st;
            double safe=d.safe;
            if(safe<o.domain_safe_floor&&d.edge>o.domain_safe_floor){
                safe=o.domain_safe_floor;Eval se{};st=eval_at(ctx,kind,safe,se);if(st!=AME_CROSSING_OK||se.outside||!isfin(se.g2)||se.g2<=o.physical_domain_margin)safe=std::nextafter(d.edge,0.0);
            }
            Eval es{};st=eval_at(ctx,kind,safe,es);if(st!=AME_CROSSING_OK)return st;
            if(!es.outside&&safe>s_prev&&crosses(prev.f,es.f,dir)){
                set_bracket(r.event_bracket,s_prev,safe,prev.f,es.f);double root;st=root_solve(ctx,kind,s_prev,safe,prev.f,es.f,o,root);if(st==AME_CROSSING_OK&&root<=safe){r.has_event=1;r.event=root;return AME_CROSSING_OK;}
            }
            r.has_domain_edge=1;r.domain_edge=d.edge;r.has_domain_safe=1;r.domain_safe=safe;set_bracket(r.domain_bracket,s_prev,s,prev.g2,e.g2);
            if(e.outside&&isfin(e.native_event)){r.has_domain_event=1;r.domain_event=e.native_event;}
            return AME_CROSSING_OK;
        }
        if(crosses(prev.f,e.f,dir)){set_bracket(r.event_bracket,s_prev,s,prev.f,e.f);double root;st=root_solve(ctx,kind,s_prev,s,prev.f,e.f,o,root);if(st!=AME_CROSSING_OK)return st;r.has_event=1;r.event=root;return AME_CROSSING_OK;}
        s_prev=s;prev=e;
    }
    return AME_CROSSING_OK;
}

// Low-degree polynomial geometry setup --------------------------------------
static double poly_eval(const std::vector<double>& c,double x){double out=0;for(auto it=c.rbegin();it!=c.rend();++it)out=std::fma(out,x,*it);return out;}
static double bisect_poly(const std::vector<double>&c,double lo,double hi){double flo=poly_eval(c,lo),fhi=poly_eval(c,hi);if(flo==0)return lo;if(fhi==0)return hi;for(int i=0;i<96;++i){double m=.5*(lo+hi);if(m==lo||m==hi)break;double fm=poly_eval(c,m);if(fm==0)return m;if(flo*fm<=0){hi=m;fhi=fm;}else{lo=m;flo=fm;}}return .5*(lo+hi);}
static std::vector<double> roots_interval(std::vector<double> c,double lo,double hi){while(c.size()>1&&c.back()==0)c.pop_back();int deg=(int)c.size()-1;if(deg<=0||!(lo<hi))return{};if(deg==1){double r=-c[0]/c[1];return(lo<=r&&r<=hi&&isfin(r))?std::vector<double>{r}:std::vector<double>{};}std::vector<double>d(deg);for(int i=0;i<deg;++i)d[i]=(i+1)*c[i+1];auto crit=roots_interval(d,lo,hi);std::vector<double>pts;pts.push_back(lo);pts.insert(pts.end(),crit.begin(),crit.end());pts.push_back(hi);std::vector<double>roots;double scale=1;for(double x:c)scale+=std::fabs(x);double ztol=256*std::nextafter(scale,pinf())-256*scale; if(!(ztol>0))ztol=256*DBL_EPSILON*scale;for(double x:crit)if(std::fabs(poly_eval(c,x))<=ztol)roots.push_back(x);for(size_t i=0;i+1<pts.size();++i){double a=pts[i],b=pts[i+1];if(!(a<b))continue;double aa=(a!=lo)?std::nextafter(a,b):a,bb=(b!=hi)?std::nextafter(b,a):b;if(!(aa<=bb))continue;double fa=poly_eval(c,aa),fb=poly_eval(c,bb);if(!isfin(fa)||!isfin(fb))continue;if(fa==0)roots.push_back(aa);if(fb==0)roots.push_back(bb);if(fa*fb<0)roots.push_back(bisect_poly(c,aa,bb));}std::sort(roots.begin(),roots.end());std::vector<double>out;for(double r:roots){double tol=64*std::nextafter(std::max(1.0,std::fabs(r)),pinf())-64*std::max(1.0,std::fabs(r));if(out.empty()||std::fabs(r-out.back())>std::fabs(tol))out.push_back(r);}return out;}

static double motor_R(double MM,double A,double B,double v){if(!(v>0))return pinf();double gap=std::fma(B,v,-A);double rad=std::fma(-gap,gap,MM*MM);if(rad<0)return qnan();return std::sqrt(std::max(0.0,rad))/(v*v);}
static double motor_Rp(double MM,double A,double B,double v){double gap=std::fma(B,v,-A),rad=std::fma(-gap,gap,MM*MM);if(!(v>0&&rad>0))return qnan();double num=std::fma(B*B,v*v,std::fma(-3*A*B,v,2*(A*A-MM*MM)));return num/(v*v*v*std::sqrt(rad));}
static double motor_H(double MM,double A,double B,double v){double rp=motor_Rp(MM,A,B,v);if(!isfin(rp)||!(v>0))return qnan();return -rp*std::fma(-B,v,A)/v;}

static ame_motor_switch_geometry_native derive_geometry(){ame_motor_switch_geometry_native g{};g.M=M;g.A=A_MAX;g.B=B_EMF;g.v_H=qnan();g.H_max=qnan();if(!(A_MAX>M))return g;g.v_min=(A_MAX-M)/B_EMF;g.v_eq=A_MAX/B_EMF;double disc=std::sqrt(std::fma(8.0,M*M,A_MAX*A_MAX));g.v_R=(3*A_MAX-disc)/(2*B_EMF);if(!(g.v_min<g.v_R&&g.v_R<g.v_eq))return g;g.R_max=motor_R(M,A_MAX,B_EMF,g.v_R);double m=M/A_MAX,m2=m*m,m4=m2*m2;std::vector<double>c={-8+16*m2-8*m4,33-39*m2+6*m4,-53+32*m2,41-9*m2,-15,2};double yR=B_EMF*g.v_R/A_MAX;auto rr=roots_interval(c,std::nextafter(yR,1.0),std::nextafter(1.0,yR));if(rr.size()==1){g.v_H=(A_MAX/B_EMF)*rr[0];g.H_max=motor_H(M,A_MAX,B_EMF,g.v_H);g.has_v_H=1;g.orientation_certified=isfin(g.H_max)&&g.H_max>0;}double ymin=(A_MAX-M)/A_MAX;auto rise=roots_interval(c,std::nextafter(ymin,yR),std::nextafter(yR,ymin));double mid=.5*(std::nextafter(ymin,yR)+std::nextafter(yR,ymin));g.rising_barrier_certified=rise.empty()&&poly_eval(c,mid)>0;return g;}
static const ame_motor_switch_geometry_native MOTOR_GEOM=derive_geometry();

static double motor_R_current(double v){return motor_R(M,A_MAX,B_EMF,v);}static double motor_H_current(double v){return motor_H(M,A_MAX,B_EMF,v);}static double motor_J(double v){return -motor_H_current(v);}

static bool decreasing_horizon(const Ctx &c,double L,double &h,bool &crosszero){if(c.impl!=AME_SEGMENT_IMPL_GRIP_CFLOW||c.sigma==0||c.k0==0)return false;double k1=std::fma(c.sigma,L,c.k0);if(k1==0||std::signbit(k1)!=std::signbit(c.k0)){double z=std::fabs(c.k0)/std::fabs(c.sigma);if(!isfin(z)||z<=0)return false;h=std::min(L,z);crosszero=true;return true;}if(std::fabs(k1)>=std::fabs(c.k0))return false;h=L;crosszero=false;return true;}

static bool motor_upper_w(double w0,double ds,double &out){ame_segment_status ss=AME_SEGMENT_OK;ame_segment_options opt=ame_segment_default_options();ame_segment *m=ame_segment_compile(ds,0.0,w0,0.0,AME_SEGMENT_MOTOR,0,&opt,&ss);if(!m)return false;double w=qnan();ss=ame_segment_w(m,ds,&w);ame_segment_destroy(m);if(ss!=AME_SEGMENT_OK||!isfin(w))return false;out=shift_ulps(w,pinf());return true;}
static double motor_R_interval_max(double vlo,double vhi){
    if(vhi<vlo)std::swap(vlo,vhi);
    double lo=std::max(vlo,MOTOR_GEOM.v_min),hi=std::min(vhi,MOTOR_GEOM.v_eq);
    if(!(lo<=hi))return 0;
    double val=0.0; bool any=false;
    for(double x : {lo,hi}){double rv=motor_R_current(x);if(isfin(rv)){val=any?std::max(val,rv):rv;any=true;}}
    if(lo<=MOTOR_GEOM.v_R&&MOTOR_GEOM.v_R<=hi&&isfin(MOTOR_GEOM.R_max)){val=any?std::max(val,MOTOR_GEOM.R_max):MOTOR_GEOM.R_max;any=true;}
    return any?shift_ulps(val,pinf()):0.0;
}

enum class DomainCert { Unknown, Inside, Domain };
static DomainCert increasing_domain_cert(const Ctx &c,double L,double margin){if(c.impl!=AME_SEGMENT_IMPL_GRIP_CFLOW||L<0||!isfin(L)||c.sigma==0||c.k0==0)return DomainCert::Unknown;double k1=std::fma(c.sigma,L,c.k0);if(k1==0||std::signbit(k1)!=std::signbit(c.k0))return DomainCert::Unknown;double r0=std::fabs(c.k0),r1=std::fabs(k1);if(!(r1>r0))return DomainCert::Unknown;double q0=r0*c.w0,g20=std::fma(-q0,q0,M2);if(!(g20>margin))return DomainCert::Domain;double g0=std::sqrt(g20);double qlo=shift_ulps(r1*c.w0,ninf()),whi=std::fma(2*g0,L,c.w0),qhi=shift_ulps(r1*whi,pinf());double qt=std::sqrt(std::max(0.0,M2-margin)),qtlo=shift_ulps(qt,ninf()),qthi=shift_ulps(qt,pinf());if(qhi<qtlo)return DomainCert::Inside;if(qlo>qthi)return DomainCert::Domain;return DomainCert::Unknown;}

static double bisect_level(double(*fn)(double),double lo,double hi,double target,bool increasing){double a=lo,b=hi;for(int i=0;i<80;++i){double m=.5*(a+b);if(m==a||m==b)break;double fm=fn(m);if((fm<target)==increasing)a=m;else b=m;}return .5*(a+b);}
static bool motor_speed_integral(double v,double v0,double &out){if(!(0<=v0&&v0<=v&&v<MOTOR_GEOM.v_eq)){if(v==v0){out=0;return true;}return false;}double gap0=std::fma(-B_EMF,v0,A_MAX);if(!(gap0>0))return false;double x=B_EMF*(v-v0)/gap0;if(!(0<=x&&x<1))return false;double rem;if(x<1e-4){double term=x*x;rem=.5*term;for(int n=3;n<9;++n){term*=x;rem+=term/n;}}else rem=-std::log1p(-x)-x;out=(A_MAX*rem+x*B_EMF*v0)/(B_EMF*B_EMF);return true;}
static bool rising_min_speed(double v0,double vhi,double slope,double &out){if(!MOTOR_GEOM.rising_barrier_certified)return false;double lo=std::max(v0,MOTOR_GEOM.v_min),hi=std::min(vhi,MOTOR_GEOM.v_R);if(!(lo<=hi))return false;double S=std::fabs(slope);if(S==0||lo==hi){out=hi;return true;}double left=std::max(lo,std::nextafter(MOTOR_GEOM.v_min,MOTOR_GEOM.v_R)),jl=motor_J(left),jh=motor_J(hi);if(!isfin(jl)||!isfin(jh))return false;if(S>=jl){out=left;return true;}if(S<=jh){out=hi;return true;}out=bisect_level(motor_J,left,hi,S,false);return true;}
static bool increasing_low_no_switch(const Ctx&c,double L){if(!MOTOR_GEOM.rising_barrier_certified)return false;double v0=std::sqrt(std::max(0.0,c.w0));if(v0>=MOTOR_GEOM.v_R||v0>=MOTOR_GEOM.v_eq)return false;double wup;if(!motor_upper_w(c.w0,L,wup)||wup<c.w0)return false;double vhi=std::min(std::sqrt(std::max(0.0,wup)),MOTOR_GEOM.v_R);if(vhi<=MOTOR_GEOM.v_min)return true;double vt;if(!rising_min_speed(v0,vhi,c.sigma,vt))return false;double integ;if(!motor_speed_integral(vt,v0,integ))return false;double rlower=std::fma(std::fabs(c.sigma),integ,std::fabs(c.k0)),R=motor_R_current(vt);return isfin(rlower)&&isfin(R)&&shift_ulps(rlower,ninf())>shift_ulps(R,pinf());}

static ame_crossing_status structural_motor_decreasing(Ctx&c,const ame_crossing_options&o,ame_crossing_result&r,bool &handled){handled=false;if(!MOTOR_GEOM.orientation_certified)return AME_CROSSING_OK;double h;bool cz;if(!decreasing_horizon(c,c.L,h,cz))return AME_CROSSING_OK;Eval e0{};auto st=eval_at(c,EvalKind::GripMotor,0,e0);if(st!=AME_CROSSING_OK)return st;if(!isfin(e0.g2)){r.has_domain_edge=1;r.domain_edge=0;set_bracket(r.domain_bracket,0,0,e0.g2,e0.g2);handled=true;return AME_CROSSING_OK;}if(e0.g2<=o.domain_margin&&!o.allow_initial_boundary){r.has_domain_edge=1;r.domain_edge=0;set_bracket(r.domain_bracket,0,0,e0.g2,e0.g2);handled=true;return AME_CROSSING_OK;}if(initial_switch_required(e0.f,e0,NEG_TO_POS,o.f_tol,locator_spatial_tol(o,c.L))){r.has_event=1;r.event=0;r.initial_switch=1;set_bracket(r.event_bracket,0,0,e0.f,e0.f);handled=true;return AME_CROSSING_OK;}if(c.L==0){handled=true;return AME_CROSSING_OK;}if(cz&&h<c.L)return AME_CROSSING_OK;double v0=std::sqrt(std::max(0.0,c.w0)),wup;if(!motor_upper_w(c.w0,h,wup)||wup<c.w0)return AME_CROSSING_OK;double vup=std::sqrt(std::max(0.0,wup)),rh=std::fabs(std::fma(c.sigma,h,c.k0)),maxR=motor_R_interval_max(v0,vup);if(shift_ulps(rh,ninf())>maxR){handled=true;return AME_CROSSING_OK;}return AME_CROSSING_OK;}

static ame_crossing_status structural_motor_increasing(Ctx&c,const ame_crossing_options&o,ame_crossing_result&r,bool &handled){handled=false;if(!MOTOR_GEOM.orientation_certified||c.impl!=AME_SEGMENT_IMPL_GRIP_CFLOW||c.sigma==0||c.k0==0||c.L<0||!isfin(c.L))return AME_CROSSING_OK;double k1=std::fma(c.sigma,c.L,c.k0);if(k1==0||std::signbit(k1)!=std::signbit(c.k0)||std::fabs(k1)<=std::fabs(c.k0))return AME_CROSSING_OK;Eval e0{};auto st=eval_at(c,EvalKind::GripMotor,0,e0);if(st!=AME_CROSSING_OK)return st;if(!isfin(e0.g2)){r.has_domain_edge=1;r.domain_edge=0;set_bracket(r.domain_bracket,0,0,e0.g2,e0.g2);handled=true;return AME_CROSSING_OK;}if(e0.g2<=o.physical_domain_margin&&!o.allow_initial_boundary){r.has_domain_edge=1;r.domain_edge=0;set_bracket(r.domain_bracket,0,0,e0.g2,e0.g2);handled=true;return AME_CROSSING_OK;}if(initial_switch_required(e0.f,e0,NEG_TO_POS,o.f_tol,locator_spatial_tol(o,c.L))){r.has_event=1;r.event=0;r.initial_switch=1;set_bracket(r.event_bracket,0,0,e0.f,e0.f);handled=true;return AME_CROSSING_OK;}if(c.L==0){handled=true;return AME_CROSSING_OK;}double v0=std::sqrt(std::max(0.0,c.w0));bool high=v0>=shift_ulps(MOTOR_GEOM.v_R,pinf()),low=false;if(!high){low=increasing_low_no_switch(c,c.L);if(!low)return AME_CROSSING_OK;}DomainCert dc=increasing_domain_cert(c,c.L,o.physical_domain_margin);if(dc==DomainCert::Inside){handled=true;return AME_CROSSING_OK;}if(dc==DomainCert::Domain&&high){r.has_domain_edge=1;r.domain_edge=c.L;r.domain_switch_excluded=1;handled=true;return AME_CROSSING_OK;}return AME_CROSSING_OK;}

static double grip_brake_gmin(const Ctx&c,double g20){if(!isfin(g20)||g20<0)return qnan();double r0=std::fabs(c.k0),S=std::fabs(c.sigma);if(r0==0||S==0)return qnan();double g0=std::sqrt(std::max(0.0,g20)),r2=r0*r0,den=std::hypot(S,2*r2);if(den==0||!isfin(den))return qnan();double gs=M*S/den;if(!isfin(gs))return qnan();return std::max(0.0,shift_ulps(std::min(g0,gs),ninf()));}
static void grip_brake_q_bounds(const Ctx&c,double h,double gmin,double &qlo,double&qhi){double r=std::fabs(std::fma(c.sigma,h,c.k0)),rlo=std::max(0.0,shift_ulps(r,ninf())),rhi=shift_ulps(r,pinf());double wlo=std::fma(2*gmin,h,c.w0),whi=std::fma(2*shift_ulps(A_BRAKE,pinf()),h,c.w0);wlo=std::max(0.0,shift_ulps(wlo,ninf()));whi=shift_ulps(whi,pinf());qlo=shift_ulps(rlo*wlo,ninf());qhi=shift_ulps(rhi*whi,pinf());}
static std::vector<double> quadratic_roots(double a,double b,double c0,double target,double L){double c=c0-target;if(a==0){if(b==0)return{};double r=-c/b;return isfin(r)&&0<=r&&r<=L?std::vector<double>{r}:std::vector<double>{};}double disc=std::fma(-4*a,c,b*b);if(disc<0||!isfin(disc))return{};double sd=std::sqrt(std::max(0.0,disc));std::vector<double>rr;if(sd==0)rr.push_back(-.5*b/a);else{double q=-.5*(b+std::copysign(sd,b));rr.push_back(q/a);if(q!=0)rr.push_back(c/q);}rr.erase(std::remove_if(rr.begin(),rr.end(),[&](double x){return!isfin(x)||x<0||x>L;}),rr.end());std::sort(rr.begin(),rr.end());if(rr.size()==2&&rr[0]==rr[1])rr.pop_back();return rr;}
static bool later_envelope_root(const Ctx&c,double h,double accel,double target,bool lower,double&root){double r0=std::fabs(c.k0),S=std::fabs(c.sigma),a=-2*S*accel,b=std::fma(2*r0,accel,-S*c.w0),c0=r0*c.w0;auto rr=quadratic_roots(a,b,c0,target,h);if(rr.empty())return false;root=shift_ulps(rr.back(),lower?ninf():pinf());root=std::min(h,std::max(0.0,root));return true;}

static ame_crossing_status structural_brake(Ctx&c,const ame_crossing_options&o,ame_crossing_result&r,bool&handled){handled=false;double h;bool cz;if(!decreasing_horizon(c,c.L,h,cz))return AME_CROSSING_OK;Eval e0{};auto st=eval_at(c,EvalKind::GripBrake,0,e0);if(st!=AME_CROSSING_OK)return st;if(!isfin(e0.g2)){r.has_domain_edge=1;r.domain_edge=0;set_bracket(r.domain_bracket,0,0,e0.g2,e0.g2);handled=true;return AME_CROSSING_OK;}if(e0.g2<=o.physical_domain_margin&&!o.allow_initial_boundary){r.has_domain_edge=1;r.domain_edge=0;set_bracket(r.domain_bracket,0,0,e0.g2,e0.g2);handled=true;return AME_CROSSING_OK;}double itol=locator_spatial_tol(o,c.L);if(initial_switch_required(e0.f,e0,NEG_TO_POS,o.f_tol,itol)){r.has_event=1;r.event=0;r.initial_switch=1;set_bracket(r.event_bracket,0,0,e0.f,e0.f);handled=true;return AME_CROSSING_OK;}if(c.L==0){handled=true;return AME_CROSSING_OK;}double g20=o.allow_initial_boundary&&e0.g2<=o.physical_domain_margin?0:e0.g2,gmin=grip_brake_gmin(c,g20);if(!isfin(gmin))return AME_CROSSING_OK;double qlo,qhi;grip_brake_q_bounds(c,h,gmin,qlo,qhi);if(!cz&&qlo>K_BRAKE_HI){handled=true;return AME_CROSSING_OK;}bool guaranteed=cz||qhi<K_BRAKE_LO;if(!guaranteed){Eval eh{};st=eval_at(c,EvalKind::GripBrake,h,eh);if(st!=AME_CROSSING_OK)return AME_CROSSING_OK;if(eh.f<0){handled=true;return AME_CROSSING_OK;}}
    double lo,hi;if(!later_envelope_root(c,h,gmin,K_BRAKE_HI,true,lo)||!later_envelope_root(c,h,shift_ulps(A_BRAKE,pinf()),K_BRAKE_LO,false,hi)||hi<lo)return AME_CROSSING_OK;Eval elo{},ehi{};if(eval_at(c,EvalKind::GripBrake,lo,elo)!=AME_CROSSING_OK||eval_at(c,EvalKind::GripBrake,hi,ehi)!=AME_CROSSING_OK)return AME_CROSSING_OK;bool depart=std::fabs(e0.f)<=o.f_tol&&e0.has_df&&e0.df<0;if(lo<=itol&&std::fabs(elo.f)<=o.f_tol&&depart){double interior=std::min(hi,std::max(itol,.125*hi));if(!(0<interior&&interior<hi))return AME_CROSSING_OK;Eval ei{};if(eval_at(c,EvalKind::GripBrake,interior,ei)!=AME_CROSSING_OK||ei.f>=0)return AME_CROSSING_OK;lo=interior;elo=ei;}if(elo.f>0||ehi.f<0)return AME_CROSSING_OK;if(elo.f==0){r.has_event=1;r.event=lo;set_bracket(r.event_bracket,lo,lo,elo.f,elo.f);handled=true;return AME_CROSSING_OK;}if(ehi.f==0){r.has_event=1;r.event=hi;set_bracket(r.event_bracket,hi,hi,ehi.f,ehi.f);handled=true;return AME_CROSSING_OK;}double guess,gguess=.5*(gmin+A_BRAKE);if(!later_envelope_root(c,h,gguess,K_BRAKE,false,guess)||!(lo<guess&&guess<hi))guess=.5*(lo+hi);Eval eg{};if(eval_at(c,EvalKind::GripBrake,guess,eg)!=AME_CROSSING_OK)return AME_CROSSING_OK;double rlo,rhi,flo,fhi;if(eg.f>=0){rlo=lo;rhi=guess;flo=elo.f;fhi=eg.f;}else{rlo=guess;rhi=hi;flo=eg.f;fhi=ehi.f;}double root;st=root_solve(c,EvalKind::GripBrake,rlo,rhi,flo,fhi,o,root);if(st!=AME_CROSSING_OK)return AME_CROSSING_OK;r.has_event=1;r.event=root;set_bracket(r.event_bracket,rlo,rhi,flo,fhi);handled=true;return AME_CROSSING_OK;}

static ame_crossing_status structural_brake_increasing(Ctx&c,const ame_crossing_options&o,ame_crossing_result&r,bool&handled){handled=false;if(c.impl!=AME_SEGMENT_IMPL_GRIP_CFLOW||c.sigma==0||c.k0==0)return AME_CROSSING_OK;double k1=std::fma(c.sigma,c.L,c.k0);if(k1==0||std::signbit(k1)!=std::signbit(c.k0)||std::fabs(k1)<=std::fabs(c.k0))return AME_CROSSING_OK;Eval e0{};auto st=eval_at(c,EvalKind::GripBrake,0,e0);if(st!=AME_CROSSING_OK)return st;if(!isfin(e0.g2)){r.has_domain_edge=1;r.domain_edge=0;set_bracket(r.domain_bracket,0,0,e0.g2,e0.g2);handled=true;return AME_CROSSING_OK;}if(e0.g2<=o.physical_domain_margin&&!o.allow_initial_boundary){r.has_domain_edge=1;r.domain_edge=0;set_bracket(r.domain_bracket,0,0,e0.g2,e0.g2);handled=true;return AME_CROSSING_OK;}if(initial_switch_required(e0.f,e0,NEG_TO_POS,o.f_tol,locator_spatial_tol(o,c.L))){r.has_event=1;r.event=0;r.initial_switch=1;set_bracket(r.event_bracket,0,0,e0.f,e0.f);handled=true;return AME_CROSSING_OK;}DomainCert dc=increasing_domain_cert(c,c.L,o.physical_domain_margin);if(dc==DomainCert::Inside){handled=true;return AME_CROSSING_OK;}if(dc==DomainCert::Domain){r.has_domain_edge=1;r.domain_edge=c.L;r.domain_switch_excluded=1;handled=true;return AME_CROSSING_OK;}return AME_CROSSING_OK;}

static double first_brake_threshold(Ctx&c,double threshold,bool event,double residual_tol){double rate=S_BRAKE,a=rate*c.sigma,b=std::fma(c.w0,c.sigma,rate*c.k0),c0=c.w0*c.k0;std::vector<double>cand;for(double target:{threshold,-threshold})for(double root:quadratic_roots(a,b,c0,target,c.L)){double w; if(ame_segment_w(c.seg,root,&w)!=AME_SEGMENT_OK)continue; ++c.evals; double k=std::fma(c.sigma,root,c.k0),q=w*k,qp=std::fma(k,rate,c.sigma*w);if(event){double g2=std::fma(-q,q,M2);if(g2<=0)continue;double g=std::sqrt(g2),res=g-A_BRAKE,der=-q*qp/g;if(der<0&&std::fabs(res)<=residual_tol)cand.push_back(root);}else if(-2*q*qp<0)cand.push_back(root);}return cand.empty()?qnan():*std::min_element(cand.begin(),cand.end());}

static ame_crossing_status brake_grip_analytic(Ctx&c,const ame_crossing_options&o,ame_crossing_result&r){Eval e0{};auto st=eval_at(c,EvalKind::BrakeGrip,0,e0);if(st!=AME_CROSSING_OK)return st;if(!isfin(e0.g2)||e0.g2<=o.domain_margin){r.has_domain_edge=1;r.domain_edge=0;set_bracket(r.domain_bracket,0,0,e0.g2,e0.g2);return AME_CROSSING_OK;}if(initial_switch_required(e0.f,e0,POS_TO_NEG,o.f_tol,locator_spatial_tol(o,c.L))){r.has_event=1;r.event=0;r.initial_switch=1;set_bracket(r.event_bracket,0,0,e0.f,e0.f);return AME_CROSSING_OK;}if(c.L==0)return AME_CROSSING_OK;double er=first_brake_threshold(c,K_BRAKE,true,std::max(1e-8,32*o.f_tol));double dt=std::sqrt(M2-o.domain_margin),dr=first_brake_threshold(c,dt,false,0);if(isfin(dr)&&(!isfin(er)||dr<=er)){r.has_domain_edge=1;r.domain_edge=dr;return AME_CROSSING_OK;}if(isfin(er)){r.has_event=1;r.event=er;}if(isfin(dr)){r.has_domain_edge=1;r.domain_edge=dr;}return AME_CROSSING_OK;}

static bool validate_options(const ame_crossing_options&o,bool safe){return o.n_scan>0&&isfin(o.domain_margin)&&o.domain_margin>=0&&isfin(o.x_abs_tol)&&o.x_abs_tol>=0&&isfin(o.x_rel_tol)&&o.x_rel_tol>=0&&isfin(o.f_tol)&&o.f_tol>=0&&o.max_iter>0&&(!o.has_initial_spatial_tol||(isfin(o.initial_spatial_tol)&&o.initial_spatial_tol>=0))&&(!safe||(o.domain_stop_margin>=o.physical_domain_margin&&o.physical_domain_margin>=0));}

} // namespace

extern "C" {

ame_crossing_options ame_crossing_default_options(void){ame_crossing_options o{};o.n_scan=256;o.domain_margin=1e-12;o.domain_stop_margin=4e-12;o.physical_domain_margin=1e-12;o.domain_safe_floor=0;o.x_abs_tol=1e-13;o.x_rel_tol=1e-13;o.f_tol=1e-13;o.max_iter=64;return o;}

ame_crossing_status ame_crossing_scan(ame_segment *segment,double L,ame_crossing_kind kind,int earliest_safe,const ame_crossing_options *options,ame_crossing_result *out){if(!segment||!out||!isfin(L)||L<0)return AME_CROSSING_INVALID_ARGUMENT;ame_crossing_options o=options?*options:ame_crossing_default_options();bool safe=earliest_safe!=0;if(!validate_options(o,safe))return AME_CROSSING_INVALID_ARGUMENT;double segL=ame_segment_length(segment);if(L>segL+64*std::numeric_limits<double>::epsilon()*std::max(1.0,std::fabs(segL)))return AME_CROSSING_INVALID_ARGUMENT;clear_result(out);Ctx c{segment,L,ame_segment_sigma(segment),ame_segment_w0(segment),ame_segment_k0(segment),ame_segment_implementation(segment),0};ame_crossing_status st=AME_CROSSING_OK;bool handled=false;
    switch(kind){
        case AME_CROSSING_MOTOR_GRIP: st=generic_scan(c,EvalKind::MotorGrip,POS_TO_NEG,o,*out);break;
        case AME_CROSSING_BRAKE_GRIP: st=brake_grip_analytic(c,o,*out);break;
        case AME_CROSSING_GRIP_MOTOR:
            if(safe){st=structural_motor_decreasing(c,o,*out,handled);if(st==AME_CROSSING_OK&&!handled)st=structural_motor_increasing(c,o,*out,handled);if(st==AME_CROSSING_OK&&!handled)st=safe_grip_scan(c,EvalKind::GripMotor,NEG_TO_POS,o,*out);}else{st=structural_motor_decreasing(c,o,*out,handled);if(st==AME_CROSSING_OK&&!handled)st=generic_scan(c,EvalKind::GripMotor,NEG_TO_POS,o,*out);}break;
        case AME_CROSSING_GRIP_BRAKE:
            st=structural_brake(c,o,*out,handled);if(st==AME_CROSSING_OK&&!handled&&safe)st=structural_brake_increasing(c,o,*out,handled);if(st==AME_CROSSING_OK&&!handled){if(safe)st=safe_grip_scan(c,EvalKind::GripBrake,NEG_TO_POS,o,*out);else st=generic_scan(c,EvalKind::GripBrake,NEG_TO_POS,o,*out);}break;
        default: st=AME_CROSSING_INVALID_ARGUMENT;break;
    }
    out->state_evaluations=c.evals;return st;}

ame_crossing_status ame_crossing_motor_grip(ame_segment*s,double L,const ame_crossing_options*o,ame_crossing_result*r){return ame_crossing_scan(s,L,AME_CROSSING_MOTOR_GRIP,0,o,r);}ame_crossing_status ame_crossing_grip_motor(ame_segment*s,double L,int safe,const ame_crossing_options*o,ame_crossing_result*r){return ame_crossing_scan(s,L,AME_CROSSING_GRIP_MOTOR,safe,o,r);}ame_crossing_status ame_crossing_grip_brake(ame_segment*s,double L,int safe,const ame_crossing_options*o,ame_crossing_result*r){return ame_crossing_scan(s,L,AME_CROSSING_GRIP_BRAKE,safe,o,r);}ame_crossing_status ame_crossing_brake_grip(ame_segment*s,double L,const ame_crossing_options*o,ame_crossing_result*r){return ame_crossing_scan(s,L,AME_CROSSING_BRAKE_GRIP,0,o,r);}
ame_crossing_status ame_crossing_motor_geometry(ame_motor_switch_geometry_native*out){if(!out)return AME_CROSSING_INVALID_ARGUMENT;*out=MOTOR_GEOM;return AME_CROSSING_OK;}
const char *ame_crossing_status_name(ame_crossing_status s){switch(s){case AME_CROSSING_OK:return"ok";case AME_CROSSING_INVALID_ARGUMENT:return"invalid_argument";case AME_CROSSING_SEGMENT_ERROR:return"segment_error";case AME_CROSSING_NUMERICAL_FAILURE:return"numerical_failure";case AME_CROSSING_UNSUPPORTED:return"unsupported";default:return"unknown";}}
}
