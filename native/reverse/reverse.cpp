#include "ame_reverse.h"
#include "../segment/include/ame_segment.h"
#include "../segment/include/ame_segment_constants.h"
#include "../crossing/include/ame_crossing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double M = AME_SEGMENT_MU_G;
constexpr double M2 = M * M;
constexpr double A_BRAKE = AME_SEGMENT_A_BRAKE;
constexpr double A_MAX = AME_SEGMENT_A_MAX;
constexpr double B_EMF = AME_SEGMENT_B_EMF;
constexpr double V_MAX = AME_SEGMENT_V_MAX;
constexpr double W_EQ = V_MAX * V_MAX;
constexpr double S_BRAKE = 2.0 * A_BRAKE;
constexpr double FRICTION_DOMAIN_MARGIN = 1.0e-12;
constexpr double INTERNAL_CAP_G2_MARGIN = 1.0e-8;
constexpr double EVENT_RESIDUAL_TOL = 1.0e-8;
constexpr double EVENT_DERIV_TOL = 1.0e-10;
constexpr double EVENT_START_MAX_SPATIAL_TOL = 1.0e-9;
constexpr double ROOT_EPS = 1.0e-11;
constexpr double PIECE_EPS = 1.0e-13;
constexpr int MAX_SUBSEGMENTS_PER_PIECE = 128;

thread_local std::string LAST_ERROR;

struct ReverseError : std::runtime_error {
    ame_reverse_status status;
    ReverseError(ame_reverse_status s, const std::string &m) : std::runtime_error(m), status(s) {}
};

static void fail(ame_reverse_status s, const std::string &m) { throw ReverseError(s, m); }
static bool isfin(double x) { return std::isfinite(x); }
static double ulp(double x) {
    if (std::isnan(x) || std::isinf(x)) return std::numeric_limits<double>::infinity();
    x = std::fabs(x);
    return std::nextafter(x, std::numeric_limits<double>::infinity()) - x;
}

/* CPython math.fsum-style partial summation, sufficient for deterministic path lengths. */
static double precise_sum(const std::vector<double> &xs) {
    std::vector<double> partials;
    partials.reserve(32);
    for (double x : xs) {
        if (!isfin(x)) return x;
        size_t j = 0;
        for (double y : partials) {
            if (std::fabs(x) < std::fabs(y)) std::swap(x, y);
            const double hi = x + y;
            const double lo = y - (hi - x);
            if (lo != 0.0) partials[j++] = lo;
            x = hi;
        }
        partials.resize(j);
        if (x != 0.0) partials.push_back(x);
    }
    double s = 0.0;
    for (auto it = partials.rbegin(); it != partials.rend(); ++it) s += *it;
    return s;
}

struct BuildWorkMetrics {
    uint64_t segment_compiles = 0;
    uint64_t crossing_calls = 0;
    uint64_t discarded_cflow_calls = 0;
    uint64_t discarded_cflow_steps = 0;
};
thread_local BuildWorkMetrics *ACTIVE_BUILD_METRICS = nullptr;

struct SegmentDeleter {
    void operator()(ame_segment *p) const {
        if (!p) return;
        if (ACTIVE_BUILD_METRICS) {
            ame_segment_cache_stats cs{};
            if (ame_segment_cache_stats_get(p, &cs) == AME_SEGMENT_OK) {
                ACTIVE_BUILD_METRICS->discarded_cflow_calls += cs.cflow_calls;
                ACTIVE_BUILD_METRICS->discarded_cflow_steps += cs.local_steps;
            }
        }
        ame_segment_destroy(p);
    }
};
using SegmentPtr = std::unique_ptr<ame_segment, SegmentDeleter>;

enum class PassKind { Forward = 1, Backward = 2 };
enum class EventKind { PieceEnd = 1, GripMotor = 2, MotorGrip = 3, GripBrake = 4, BrakeGrip = 5 };
using Mode = ame_segment_mode;

struct TraversalPiece {
    double L = 0.0;
    double sigma = 0.0;
    int piece_index = 0;
    double abs0 = 0.0;
    double direction = 1.0;
};

struct ScalarSegment {
    PassKind kind = PassKind::Forward;
    Mode mode = AME_SEGMENT_GRIP;
    EventKind event = EventKind::PieceEnd;
    int traversal_index = 0;
    int piece_index = 0;
    double offset0 = 0.0;
    double L_used = 0.0;
    double sigma = 0.0;
    double abs0 = 0.0;
    double abs1 = 0.0;
    double direction = 1.0;
    double w0 = 0.0;
    double k0 = 0.0;
    double w1 = 0.0;
    double k1 = 0.0;
    bool boundary_start = false;
    SegmentPtr seg;
};

struct ScalarPass {
    PassKind kind = PassKind::Forward;
    std::vector<TraversalPiece> pieces;
    std::vector<ScalarSegment> segments;
    double final_w = 0.0;
    double final_k = 0.0;
    Mode final_mode = AME_SEGMENT_GRIP;
    bool terminated_at_domain = false;
    int initial_knot_index = -1;
    double initial_w_dk = 0.0;
};

struct EnvelopePiece {
    PassKind source = PassKind::Forward;
    int source_index = 0;
    double abs0 = 0.0;
    double abs1 = 0.0;
    double local0 = 0.0;
    double local1 = 0.0;
    int pass_index = 0;
};

struct InternalCapAnchor {
    int knot_index = 0;
    double station = 0.0;
    double signed_k = 0.0;
    double cap_w = 0.0;
    double initial_w_dk = 0.0;
};
struct AnchorWitness { double abs0=0, abs1=0; int kind=0; double magnitude=0; };
struct AnchorStats {
    size_t possible_anchors=0;
    std::vector<int> inserted;
    size_t rounds=0, coverage_witnesses=0, continuity_witnesses=0;
};

struct ScalarBuildImpl {
    std::vector<double> raw;
    std::vector<TraversalPiece> forward_pieces;
    std::vector<TraversalPiece> backward_pieces;
    std::vector<ScalarPass> passes;
    std::vector<EnvelopePiece> envelope;
    AnchorStats anchor_stats;
    ame_reverse_options options{};
    BuildWorkMetrics build_work{};
    uint64_t build_cflow_calls = 0;
    uint64_t build_cflow_steps = 0;
};

struct RevSegment {
    ScalarSegment *scalar = nullptr;
    SegmentPtr seg;
    ame_segment_state_jac endpoint_state{};
    bool has_endpoint_state = false;
    bool has_event_cols = false;
    std::array<double,4> F_cols{};
    double F_L = 0.0;
};
struct RevPass {
    ScalarPass *scalar_pass = nullptr;
    std::vector<RevSegment> segments;
    int n_pieces = 0;
};
struct ReverseBuildImpl {
    ScalarBuildImpl *scalar = nullptr;
    bool time_only = false;
    std::vector<RevPass> passes;
};

struct SegmentExtraAdj { double aL=0, aw0=0, ak0=0, asigma=0, aabs0=0; };

struct RawGradientAccumulator {
    int n=0;
    std::vector<double> grad;
    std::vector<double> prefix;
    explicit RawGradientAccumulator(int n_) : n(n_), grad(2*n_,0.0), prefix(n_+1,0.0) {}
    void add_length(int i,double a){ if(a!=0) grad[2*i]+=a; }
    void add_raw_sigma(int i,double a){ if(a!=0) grad[2*i+1]+=a; }
    void add_traversal_sigma(const ScalarSegment&s,double a){ if(a==0)return; add_raw_sigma(s.piece_index, s.kind==PassKind::Forward?a:-a); }
    void add_prefix_exclusive(int i,double a){ if(a==0||i<=0)return; prefix[0]+=a; prefix[i]-=a; }
    void add_prefix_inclusive(int i,double a){ if(a==0)return; int e=i+1; if(e<=0||e>n) fail(AME_REVERSE_NUMERICAL_FAILURE,"prefix inclusive index"); prefix[0]+=a; prefix[e]-=a; }
    void add_abs0(const ScalarSegment&s,double a){ if(a==0)return; if(s.kind==PassKind::Forward)add_prefix_exclusive(s.piece_index,a);else add_prefix_inclusive(s.piece_index,a); }
    void add_global_end(double a){ if(a==0||n==0)return; prefix[0]+=a; prefix[n]-=a; }
    std::vector<double> finalize(){ auto out=grad; double run=0; for(int i=0;i<n;++i){run+=prefix[i];out[2*i]+=run;} return out; }
};

static double friction_g2(double w,double k){ double q=w*k; return std::fma(-q,q,M2); }
static bool friction_domain_ok(double w,double k,double margin){ if(!isfin(w)||!isfin(k)||w<=0)return false;double g=friction_g2(w,k);return isfin(g)&&g>margin; }
static void require_domain(double w,double k,double margin,const char*where){ if(!friction_domain_ok(w,k,margin))fail(AME_REVERSE_DOMAIN,std::string("friction domain: ")+where); }

static ame_reverse_mode public_mode(Mode m){ return static_cast<ame_reverse_mode>(m); }
static ame_reverse_pass_kind public_pass(PassKind k){ return k==PassKind::Forward?AME_REVERSE_PASS_FORWARD:AME_REVERSE_PASS_BACKWARD; }
static ame_reverse_event_kind public_event(EventKind e){ return static_cast<ame_reverse_event_kind>(e); }

static std::vector<double> validate_raw(const double *p,size_t n){
    if(!p||n==0||n%2) fail(AME_REVERSE_INVALID_ARGUMENT,"raw parameter array must be nonempty and even");
    std::vector<double> raw(p,p+n);
    for(size_t i=0;i<n;i+=2){ if(!isfin(raw[i])||raw[i]<=0)fail(AME_REVERSE_INVALID_ARGUMENT,"segment length invalid"); if(!isfin(raw[i+1]))fail(AME_REVERSE_INVALID_ARGUMENT,"segment sigma invalid"); }
    return raw;
}
static double total_length(const std::vector<double>&raw){ std::vector<double> ls;ls.reserve(raw.size()/2);for(size_t i=0;i<raw.size();i+=2)ls.push_back(raw[i]);return precise_sum(ls); }
static double geometry_endpoint_k(const std::vector<double>&raw,double initial_k){ double k=initial_k;for(size_t i=0;i<raw.size()/2;++i)k=std::fma(raw[2*i+1],raw[2*i],k);return k; }
static std::vector<TraversalPiece> forward_traversal(const std::vector<double>&raw){std::vector<TraversalPiece>o;o.reserve(raw.size()/2);double s=0;for(size_t i=0;i<raw.size()/2;++i){o.push_back({raw[2*i],raw[2*i+1],(int)i,s,1});s+=raw[2*i];}return o;}
static std::vector<TraversalPiece> backward_traversal(const std::vector<double>&raw){size_t n=raw.size()/2;std::vector<double> starts(n);double s=0;for(size_t i=0;i<n;++i){starts[i]=s;s+=raw[2*i];}std::vector<TraversalPiece>o;o.reserve(n);for(size_t jj=0;jj<n;++jj){size_t i=n-1-jj;o.push_back({raw[2*i],-raw[2*i+1],(int)i,starts[i]+raw[2*i],-1});}return o;}

static double local_dw(Mode mode, ame_segment *seg,double ds){ double w;auto st=ame_segment_w(seg,ds,&w);if(st!=AME_SEGMENT_OK)fail(AME_REVERSE_NUMERICAL_FAILURE,"segment w in local_dw");if(mode==AME_SEGMENT_MOTOR)return 2.0*std::fma(B_EMF,-std::sqrt(w),A_MAX);if(mode==AME_SEGMENT_BRAKE)return S_BRAKE;double k=std::fma(ame_segment_sigma(seg),ds,ame_segment_k0(seg));double g=friction_g2(w,k);if(!(g>0))fail(AME_REVERSE_DOMAIN,"grip derivative outside domain");return 2*std::sqrt(g);}
static std::pair<double,double> segment_end_state(ame_segment*seg,double ds){double w;auto st=ame_segment_w(seg,ds,&w);if(st!=AME_SEGMENT_OK)fail(st==AME_SEGMENT_DOMAIN?AME_REVERSE_DOMAIN:AME_REVERSE_NUMERICAL_FAILURE,"segment endpoint state");return {w,std::fma(ame_segment_sigma(seg),ds,ame_segment_k0(seg))};}
static double abs_to_local(const ScalarSegment&r,double s){double ds=r.direction*(s-r.abs0);double mx=std::max({std::fabs(s),std::fabs(r.abs0),std::fabs(r.abs1),std::fabs(r.L_used),1.0});double tol=32*ulp(mx);if(ds<0&&ds>=-tol)return 0;if(ds>r.L_used&&ds<=r.L_used+tol)return r.L_used;return ds;}
static std::pair<double,double> interval(const ScalarSegment&r){return {std::min(r.abs0,r.abs1),std::max(r.abs0,r.abs1)};}
static double segment_w_abs(ScalarSegment&r,double s){double w;auto st=ame_segment_w(r.seg.get(),abs_to_local(r,s),&w);if(st!=AME_SEGMENT_OK)fail(AME_REVERSE_NUMERICAL_FAILURE,"envelope segment w");return w;}
static double segment_abs_dw(ScalarSegment&r,double s){double ds=abs_to_local(r,s);return r.direction*local_dw(r.mode,r.seg.get(),ds);}

static std::pair<double,double> event_partials(EventKind e,double w,double k){double q=w*k,g2=std::fma(-q,q,M2);if(!(g2>0))fail(AME_REVERSE_DOMAIN,"event partial at domain");double G=std::sqrt(g2);if(e==EventKind::GripMotor||e==EventKind::MotorGrip){double y=std::sqrt(std::max(0.0,w));if(!(y>0))fail(AME_REVERSE_NUMERICAL_FAILURE,"motor event partial singular");return { -q*k/G+B_EMF/(2*y), -q*w/G};}if(e==EventKind::GripBrake||e==EventKind::BrakeGrip)return {-q*k/G,-q*w/G};fail(AME_REVERSE_NUMERICAL_FAILURE,"bad event kind"); return {}; }
static std::tuple<double,double,double> event_residual(EventKind e,Mode mode,ame_segment*seg,double ds){double w;auto st=ame_segment_w(seg,ds,&w);if(st!=AME_SEGMENT_OK)return {NAN,NAN,-INFINITY};double k=std::fma(ame_segment_sigma(seg),ds,ame_segment_k0(seg));double g2=friction_g2(w,k);if(!(w>0&&g2>0&&isfin(g2)))return {NAN,NAN,g2};double G=std::sqrt(g2);double F=(e==EventKind::GripMotor||e==EventKind::MotorGrip)?std::fma(B_EMF,std::sqrt(w),G-A_MAX):G-A_BRAKE;auto [Fw,Fk]=event_partials(e,w,k);double dF=std::fma(Fw,local_dw(mode,seg,ds),Fk*ame_segment_sigma(seg));return {F,dF,g2};}
static bool validate_root(EventKind e,Mode mode,ame_segment*seg,double r,double margin){if(!isfin(r))return false;auto[F,dF,g2]=event_residual(e,mode,seg,r);if(!isfin(F)||!isfin(dF)||!isfin(g2)||g2<=margin||std::fabs(dF)<=EVENT_DERIV_TOL||std::fabs(F)>EVENT_RESIDUAL_TOL)return false;bool up=(e==EventKind::GripMotor||e==EventKind::GripBrake);return up?dF>EVENT_DERIV_TOL:dF<-EVENT_DERIV_TOL;}
static double coordinate_resolution(double value,double derivative){if(!isfin(value)||!isfin(derivative)||derivative==0)return INFINITY;double target=std::nextafter(value,derivative>0?INFINITY:-INFINITY);double d=std::fabs(target-value);if(!isfin(d)||d<=0)return INFINITY;double s=d/std::fabs(derivative);return isfin(s)&&s>0?s:INFINITY;}
static double initial_event_tol(EventKind e,Mode mode,ame_segment*seg,double L){double w0=ame_segment_w0(seg),k0=ame_segment_k0(seg),sigma=ame_segment_sigma(seg),dw=0;if(mode==AME_SEGMENT_MOTOR)dw=2*std::fma(B_EMF,-std::sqrt(std::max(0.0,w0)),A_MAX);else if(mode==AME_SEGMENT_GRIP)dw=2*std::sqrt(std::max(0.0,friction_g2(w0,k0)));else dw=S_BRAKE;double cr=std::min(coordinate_resolution(w0,dw),coordinate_resolution(k0,sigma));if(!isfin(cr))cr=0;double rr=0;try{auto[F,dF,g2]=event_residual(e,mode,seg,0);(void)g2;auto[Fw,Fk]=event_partials(e,w0,k0);if(isfin(F)&&isfin(dF)&&dF!=0){std::vector<double> terms={std::fabs(Fw)*ulp(w0),std::fabs(Fk)*ulp(k0),8*ulp(std::max({std::fabs(F),A_MAX,A_BRAKE,1.0}))};rr=precise_sum(terms)/std::fabs(dF);}}catch(...){rr=0;}double r=std::max(cr,rr);if(!isfin(r)||r<=0)return 0;return std::min(r,EVENT_START_MAX_SPATIAL_TOL*std::max(1.0,std::fabs(L)));}

static Mode choose_initial(PassKind kind,double w,double k,double margin){require_domain(w,k,margin,"choose initial");double G=std::sqrt(friction_g2(w,k));if(kind==PassKind::Forward){double h=std::fma(B_EMF,std::sqrt(w),G-A_MAX);return (w<W_EQ&&h>=0)?AME_SEGMENT_MOTOR:AME_SEGMENT_GRIP;}double h=G-A_BRAKE;return h>=0?AME_SEGMENT_BRAKE:AME_SEGMENT_GRIP;}
struct CrossingSpec {EventKind event;Mode next;ame_crossing_kind kind;};
static CrossingSpec crossing_spec(PassKind kind,Mode mode){if(kind==PassKind::Forward&&mode==AME_SEGMENT_MOTOR)return{EventKind::MotorGrip,AME_SEGMENT_GRIP,AME_CROSSING_MOTOR_GRIP};if(kind==PassKind::Forward&&mode==AME_SEGMENT_GRIP)return{EventKind::GripMotor,AME_SEGMENT_MOTOR,AME_CROSSING_GRIP_MOTOR};if(kind==PassKind::Backward&&mode==AME_SEGMENT_BRAKE)return{EventKind::BrakeGrip,AME_SEGMENT_GRIP,AME_CROSSING_BRAKE_GRIP};if(kind==PassKind::Backward&&mode==AME_SEGMENT_GRIP)return{EventKind::GripBrake,AME_SEGMENT_BRAKE,AME_CROSSING_GRIP_BRAKE};fail(AME_REVERSE_TOPOLOGY_FAILURE,"invalid mode for pass"); return {}; }

static SegmentPtr compile_segment_native(Mode mode,double L,double sigma,double w0,double k0,bool grad,bool boundary=false,bool reverse_eta=false,bool has_auth=false,double auth=0){if(ACTIVE_BUILD_METRICS)++ACTIVE_BUILD_METRICS->segment_compiles;ame_segment_options o=ame_segment_default_options();o.boundary_start=boundary?1:0;o.reverse_eta=reverse_eta?1:0;o.has_authoritative_w1=has_auth?1:0;o.authoritative_w1=auth;ame_segment_status st=AME_SEGMENT_OK;ame_segment*p=ame_segment_compile(L,sigma,w0,k0,mode,grad?1:0,&o,&st);if(!p){ame_reverse_status rs=(st==AME_SEGMENT_DOMAIN)?AME_REVERSE_DOMAIN:(st==AME_SEGMENT_CONDITIONING?AME_REVERSE_CONDITIONING:AME_REVERSE_NUMERICAL_FAILURE);fail(rs,std::string("segment compile: ")+ame_segment_status_name(st));}return SegmentPtr(p);}

struct DomainClip { double safe=0,edge=0,safe_g2=0,edge_g2=0; };
static double g2_or_neg_inf(ame_segment*seg,double ds){ame_segment_crossing_state s{};auto st=ame_segment_crossing_state_at(seg,ds,&s);if(st!=AME_SEGMENT_OK||s.outside_domain||!isfin(s.g2))return -INFINITY;return s.g2;}
static double replay_target(double margin){return std::max(4*margin,128*ulp(M2));}
static bool first_domain_clip(ame_segment*seg,double L,double margin,int n_scan,DomainClip&out){if(L<=0)return false;double gp=g2_or_neg_inf(seg,0);if(gp<=margin){out={0,0,gp,gp};return true;}double sp=0;for(int j=1;j<=n_scan;++j){double s=L*(double(j)/n_scan),g=g2_or_neg_inf(seg,s);if(g<=margin){double lo=sp,hi=s,glo=gp,ghi=g;for(int i=0;i<96;++i){double m=.5*(lo+hi);if(m==lo||m==hi)break;double gm=g2_or_neg_inf(seg,m);if(gm>margin){lo=m;glo=gm;}else{hi=m;ghi=gm;}}double target=replay_target(margin);if(gp>=target&&glo<target){double a=sp,b=lo,ga=gp;for(int i=0;i<96;++i){double m=.5*(a+b);if(m==a||m==b)break;double gm=g2_or_neg_inf(seg,m);if(gm>=target){a=m;ga=gm;}else b=m;}lo=a;glo=ga;}out={lo,hi,glo,ghi};return true;}sp=s;gp=g;}return false;}

struct Probe { SegmentPtr seg; double length=0; bool domain_limited=false; double event_position=NAN; int cflow_status=0; };
static Probe compile_eval_probe(Mode mode,double L,double sigma,double w0,double k0,double min_prefix,double margin,int domain_scan,bool boundary){
    try{
        SegmentPtr seg=compile_segment_native(mode,L,sigma,w0,k0,false,boundary);
        if(mode!=AME_SEGMENT_GRIP||ame_segment_implementation(seg.get())!=AME_SEGMENT_IMPL_GRIP_CFLOW)return {std::move(seg),L,false,NAN,0};
        ame_segment_domain_probe pr{};auto st=ame_segment_domain_probe_at(seg.get(),L,&pr);if(st!=AME_SEGMENT_OK)fail(st==AME_SEGMENT_CONDITIONING?AME_REVERSE_CONDITIONING:AME_REVERSE_NUMERICAL_FAILURE,"domain probe");if(pr.pre_event)return{std::move(seg),L,false,NAN,pr.cflow_status};double event=pr.event_position;if(!isfin(event)||!(event>0&&event<=L))fail(AME_REVERSE_NUMERICAL_FAILURE,"bad Cflow event");DomainClip clip{};bool has=first_domain_clip(seg.get(),event,margin,domain_scan,clip);double floor=std::max(min_prefix,64*ulp(std::max(1.0,L)));double safe=(has&&clip.safe>0)?clip.safe:.5*event;if(safe<floor&&event>floor)safe=floor;if(!(safe<event))safe=std::nextafter(event,0.0);return{std::move(seg),safe,true,event,pr.cflow_status};
    }catch(const ReverseError&e){
        if(e.status!=AME_REVERSE_DOMAIN||mode!=AME_SEGMENT_GRIP||sigma!=0||boundary)throw;
        double ka=std::fabs(k0);if(!(ka>0))throw;double z=ka*w0/M;if(!(z>=0&&z<=1))throw;double event=(M_PI_2-std::asin(z))/(2*ka);if(!(isfin(event)&&event>0&&event<=L))throw;SegmentPtr es=compile_segment_native(mode,event,sigma,w0,k0,false,false);DomainClip clip{};bool has=first_domain_clip(es.get(),event,margin,domain_scan,clip);double floor=std::max(min_prefix,64*ulp(std::max(1.0,L)));double safe=(has&&clip.safe>0)?clip.safe:.5*event;if(safe<floor&&event>floor)safe=floor;if(!(safe<event))safe=std::nextafter(event,0.0);SegmentPtr ss=compile_segment_native(mode,safe,sigma,w0,k0,false,false);return{std::move(ss),safe,true,event,0};
    }
}

static std::array<double,4> local_event_columns(ame_segment*seg,double ds,EventKind event,const ame_segment_state_jac*known=nullptr){ame_segment_state_jac sj{};if(known)sj=*known;else{auto st=ame_segment_w_and_jac(seg,ds,&sj);if(st!=AME_SEGMENT_OK)fail(AME_REVERSE_NUMERICAL_FAILURE,"event jac");}double k=std::fma(ame_segment_sigma(seg),ds,ame_segment_k0(seg));auto[Fw,Fk]=event_partials(event,sj.w,k);std::array<double,4>c{};for(int i=0;i<4;++i)c[i]=std::fma(Fw,sj.jac[i],Fk*sj.jac[4+i]);return c;}

static ame_crossing_result run_crossing(ame_segment *seg,double L,const CrossingSpec &spec,bool safe,int n_scan,double margin,double safe_floor,bool boundary,double init_tol){
    if(ACTIVE_BUILD_METRICS)++ACTIVE_BUILD_METRICS->crossing_calls;
    ame_crossing_options o=ame_crossing_default_options();o.n_scan=n_scan;o.domain_margin=margin;o.physical_domain_margin=margin;o.domain_stop_margin=replay_target(margin);o.domain_safe_floor=safe_floor;o.allow_initial_boundary=boundary?1:0;o.has_initial_spatial_tol=1;o.initial_spatial_tol=init_tol;
    ame_crossing_result r{};ame_crossing_status st=ame_crossing_scan(seg,L,spec.kind,safe?1:0,&o,&r);if(st!=AME_CROSSING_OK)fail(st==AME_CROSSING_SEGMENT_ERROR?AME_REVERSE_NUMERICAL_FAILURE:AME_REVERSE_TOPOLOGY_FAILURE,std::string("crossing scan: ")+ame_crossing_status_name(st));return r;
}

static ScalarPass build_scalar_pass(const std::vector<TraversalPiece>&pieces,double init_w,double init_k,PassKind kind,const ame_reverse_options&o,int initial_knot_index=-1,double initial_w_dk=0.0,Mode init_mode=static_cast<Mode>(0)){
    require_domain(init_w,init_k,o.domain_margin,"scalar pass initial");
    ScalarPass pass;pass.kind=kind;pass.pieces=pieces;pass.initial_knot_index=initial_knot_index;pass.initial_w_dk=initial_w_dk;pass.segments.reserve(pieces.size()*2+4);
    double w=init_w,k=init_k;Mode mode=init_mode?init_mode:choose_initial(kind,w,k,o.domain_margin);bool boundary_pending=initial_knot_index>=0;int shared_scan=std::max(o.n_scan,o.domain_scan);
    for(size_t ti=0;ti<pieces.size();++ti){const auto &piece=pieces[ti];double offset=0;
        int seg_iter=0;for(;seg_iter<MAX_SUBSEGMENTS_PER_PIECE;++seg_iter){double remaining=piece.L-offset;if(remaining<=PIECE_EPS)break;
            bool boundary_here=boundary_pending&&ti==0&&offset==0&&mode==AME_SEGMENT_GRIP&&k!=0&&piece.sigma/k<0;
            if(boundary_here)w=M/std::fabs(k);else require_domain(w,k,o.domain_margin,"scalar segment start");
            CrossingSpec spec=crossing_spec(kind,mode);double min_prefix=std::max(PIECE_EPS,1e-14*std::max(1.0,remaining));bool fused=o.fused_grip_discovery&&mode==AME_SEGMENT_GRIP&&piece.sigma!=0;
            SegmentPtr seg;Probe probe{};double probe_L=remaining;ame_crossing_result scan{};
            if(fused){
                try{seg=compile_segment_native(mode,remaining,piece.sigma,w,k,false,boundary_here);}catch(const ReverseError&){fused=false;}
            }
            if(fused){
                double itol=initial_event_tol(spec.event,mode,seg.get(),remaining);scan=run_crossing(seg.get(),remaining,spec,true,shared_scan,o.domain_margin,min_prefix,boundary_here,itol);
                if(boundary_here)boundary_pending=false;
                if(!scan.has_event&&!scan.has_domain_edge){ame_segment_cache_stats cs{};if(ame_segment_cache_stats_get(seg.get(),&cs)==AME_SEGMENT_OK&&cs.local_steps>=20000){ame_segment_domain_probe pr{};auto st=ame_segment_domain_probe_at(seg.get(),remaining,&pr);if(st!=AME_SEGMENT_OK)fail(st==AME_SEGMENT_CONDITIONING?AME_REVERSE_CONDITIONING:AME_REVERSE_NUMERICAL_FAILURE,"conditioning replay");}}
                if(scan.has_domain_edge&&!scan.has_event){
                    probe=compile_eval_probe(mode,remaining,piece.sigma,w,k,min_prefix,o.domain_margin,shared_scan,boundary_here);seg=std::move(probe.seg);probe_L=probe.length;
                    if(scan.domain_switch_excluded){std::memset(&scan,0,sizeof(scan));}
                    else {double itol2=initial_event_tol(spec.event,mode,seg.get(),probe_L);scan=run_crossing(seg.get(),probe_L,spec,false,shared_scan,o.domain_margin,0,boundary_here,itol2);}
                    fused=false;
                }
            } else {
                probe=compile_eval_probe(mode,remaining,piece.sigma,w,k,min_prefix,o.domain_margin,shared_scan,boundary_here);seg=std::move(probe.seg);probe_L=probe.length;double itol=initial_event_tol(spec.event,mode,seg.get(),probe_L);scan=run_crossing(seg.get(),probe_L,spec,false,shared_scan,o.domain_margin,0,boundary_here,itol);if(boundary_here)boundary_pending=false;
            }
            if(scan.initial_switch){mode=spec.next;continue;}
            bool has_r=scan.has_event&&isfin(scan.event)&&validate_root(spec.event,mode,seg.get(),scan.event,o.domain_margin);double r=has_r?scan.event:NAN;bool event_limited=has_r&&r>0&&r<remaining-PIECE_EPS;bool boundary_event=has_r&&r>=remaining-PIECE_EPS&&r<=remaining+ROOT_EPS;bool has_edge=scan.has_domain_edge;double edge=scan.domain_edge;DomainClip recovered{};bool has_recovered=false;
            if(scan.has_event&&!has_r&&!has_edge){has_recovered=first_domain_clip(seg.get(),probe_L,o.domain_margin,shared_scan,recovered);if(has_recovered){has_edge=true;edge=recovered.edge;}}
            bool domain_before=has_edge&&(!event_limited||edge<=r+std::max(ROOT_EPS,PIECE_EPS));bool compiler_stop=(!fused)&&probe.domain_limited&&!has_r;bool terminate=domain_before||compiler_stop;
            if(has_r&&r<=0){mode=spec.next;continue;}
            double L_used;EventKind used_event;Mode mode_after;
            if(terminate){
                if(fused&&scan.has_domain_safe){
                    if(scan.has_domain_event){
                        SegmentPtr cert=compile_segment_native(mode,remaining,piece.sigma,w,k,false,boundary_here);ame_segment_domain_probe pr{};auto st=ame_segment_domain_probe_at(cert.get(),std::min(remaining,edge),&pr);if(st!=AME_SEGMENT_OK)fail(AME_REVERSE_NUMERICAL_FAILURE,"domain recert probe");double ce=pr.event_position;if(pr.pre_event||!isfin(ce)){L_used=std::min(probe_L,scan.domain_safe);}else{DomainClip cc{};bool has=first_domain_clip(cert.get(),ce,o.domain_margin,shared_scan,cc);double floor=std::max(min_prefix,64*ulp(std::max(1.0,remaining)));double safe=(has&&cc.safe>0)?cc.safe:.5*ce;if(safe<floor&&ce>floor)safe=floor;if(!(safe<ce))safe=std::nextafter(ce,0.0);L_used=std::min(probe_L,safe);}
                    }else L_used=std::min(probe_L,scan.domain_safe);
                } else if(!has_edge)L_used=probe_L;
                else {DomainClip cl{};if(has_recovered)cl=recovered;else if(!first_domain_clip(seg.get(),std::min(probe_L,edge),o.domain_margin,shared_scan,cl))fail(AME_REVERSE_DOMAIN,"domain edge recertification");L_used=std::min(probe_L,cl.safe);}
                if(L_used<=PIECE_EPS){pass.final_w=w;pass.final_k=k;pass.final_mode=mode;pass.terminated_at_domain=true;return pass;}
                used_event=EventKind::PieceEnd;mode_after=mode;
            } else if(event_limited){L_used=r;used_event=spec.event;mode_after=spec.next;}
            else {L_used=remaining;used_event=EventKind::PieceEnd;mode_after=boundary_event?spec.next:mode;}
            auto[w1,k1]=segment_end_state(seg.get(),L_used);
            if(!friction_domain_ok(w1,k1,o.domain_margin)){DomainClip cl{};if(!first_domain_clip(seg.get(),L_used,o.domain_margin,std::max(shared_scan,2*o.domain_scan),cl))fail(AME_REVERSE_DOMAIN,"emitted endpoint domain failure");L_used=cl.safe;terminate=true;used_event=EventKind::PieceEnd;mode_after=mode;if(L_used<=PIECE_EPS){pass.final_w=w;pass.final_k=k;pass.final_mode=mode;pass.terminated_at_domain=true;return pass;}std::tie(w1,k1)=segment_end_state(seg.get(),L_used);}
            require_domain(w1,k1,o.domain_margin,"emitted endpoint");double abs0=piece.abs0+piece.direction*offset,abs1=piece.abs0+piece.direction*(offset+L_used);ScalarSegment rec;rec.kind=kind;rec.mode=mode;rec.event=used_event;rec.traversal_index=(int)ti;rec.piece_index=piece.piece_index;rec.offset0=offset;rec.L_used=L_used;rec.sigma=piece.sigma;rec.abs0=abs0;rec.abs1=abs1;rec.direction=piece.direction;rec.w0=w;rec.k0=k;rec.w1=w1;rec.k1=k1;rec.boundary_start=boundary_here;rec.seg=std::move(seg);pass.segments.push_back(std::move(rec));
            offset+=L_used;w=w1;k=k1;if(terminate){pass.final_w=w;pass.final_k=k;pass.final_mode=mode;pass.terminated_at_domain=true;return pass;}if(used_event==EventKind::PieceEnd){mode=mode_after;break;}mode=mode_after;
        }
        if(seg_iter>=MAX_SUBSEGMENTS_PER_PIECE)fail(AME_REVERSE_TOPOLOGY_FAILURE,"switching chatter");
    }
    pass.final_w=w;pass.final_k=k;pass.final_mode=mode;return pass;
}

/* Safeguarded root solve used only for scalar-envelope intersections. */
template<class F> static bool bracketed_root(F &&fn,double lo,double hi,double flo,double fhi,double &root){if(hi<lo){std::swap(lo,hi);std::swap(flo,fhi);}if(flo==0){root=lo;return true;}if(fhi==0){root=hi;return true;}if(!isfin(flo)||!isfin(fhi)||flo*fhi>0)return false;double x=(fhi!=flo)?hi-fhi*(hi-lo)/(fhi-flo):.5*(lo+hi);if(!(lo<x&&x<hi)||!isfin(x))x=.5*(lo+hi);for(int i=0;i<64;++i){auto [fx,df]=fn(x);if(!isfin(fx)){x=.5*(lo+hi);std::tie(fx,df)=fn(x);if(!isfin(fx))return false;}if(std::fabs(fx)<=1e-13){root=x;return true;}if(flo*fx<=0){hi=x;fhi=fx;}else{lo=x;flo=fx;}double mid=.5*(lo+hi);if(hi-lo<=1e-13+1e-13*std::max(1.0,std::fabs(mid))){root=mid;return true;}if(isfin(df)&&df!=0){double xn=x-fx/df;if(lo<xn&&xn<hi&&isfin(xn)){x=xn;continue;}}if(fhi!=flo){double xs=hi-fhi*(hi-lo)/(fhi-flo);if(lo<xs&&xs<hi&&isfin(xs)){x=xs;continue;}}x=mid;}root=.5*(lo+hi);return true;}

static std::pair<double,double> candidate_diff(ScalarSegment&a,ScalarSegment&b,double s,bool deriv){double v=segment_w_abs(a,s)-segment_w_abs(b,s);if(!deriv)return{v,NAN};return{v,segment_abs_dw(a,s)-segment_abs_dw(b,s)};}
static bool candidate_root(ScalarSegment&a,ScalarSegment&b,double lo,double hi,double flo,double fhi,double&root){auto fn=[&](double s){return candidate_diff(a,b,s,true);};return bracketed_root(fn,lo,hi,flo,fhi,root);}
static void append_env(std::vector<EnvelopePiece>&out,int pass_index,int seg_index,ScalarSegment&r,double a,double b,double eps){if(b<=a+eps)return;double l0=abs_to_local(r,a),l1=abs_to_local(r,b);if(!out.empty()){auto&last=out.back();if(last.pass_index==pass_index&&last.source_index==seg_index&&std::fabs(last.abs1-a)<=eps){last.abs1=b;last.local1=l1;return;}}out.push_back({r.kind,seg_index,a,b,l0,l1,pass_index});}

static std::vector<EnvelopePiece> merge_candidate_envelope(std::vector<ScalarPass>&passes,double total_L,double eps=1e-12){struct Ref{double lo,hi;int p,s;};std::vector<Ref>refs;std::vector<double>cuts={0,total_L};for(size_t p=0;p<passes.size();++p)for(size_t s=0;s<passes[p].segments.size();++s){auto[lo,hi]=interval(passes[p].segments[s]);if(hi<=lo+eps)continue;refs.push_back({lo,hi,(int)p,(int)s});cuts.push_back(lo);cuts.push_back(hi);}std::sort(cuts.begin(),cuts.end());std::vector<double>uc;for(double x:cuts){x=std::min(std::max(x,0.0),total_L);if(uc.empty()||x>uc.back()+eps)uc.push_back(x);else if(x>uc.back())uc.back()=x;}std::vector<EnvelopePiece>out;for(size_t ci=0;ci+1<uc.size();++ci){double blo=uc[ci],bhi=uc[ci+1];if(bhi<=blo+eps)continue;double mid=.5*(blo+bhi);std::vector<Ref>active;for(auto&r:refs)if(r.lo<=mid&&mid<=r.hi)active.push_back(r);if(active.empty())fail(AME_REVERSE_TOPOLOGY_FAILURE,"candidate profile coverage gap");std::vector<double>roots;for(size_t i=0;i<active.size();++i){auto &ai=active[i];auto &ri=passes[ai.p].segments[ai.s];for(size_t j=i+1;j<active.size();++j){auto &aj=active[j];auto&rj=passes[aj.p].segments[aj.s];double fl=candidate_diff(ri,rj,blo,false).first,fh=candidate_diff(ri,rj,bhi,false).first;if(!isfin(fl)||!isfin(fh))continue;double root;bool has=false;if(fl==0){root=blo;has=true;}else if(fh==0){root=bhi;has=true;}else if(fl*fh<0)has=candidate_root(ri,rj,blo,bhi,fl,fh,root);if(has&&root>blo+eps&&root<bhi-eps)roots.push_back(root);}}std::sort(roots.begin(),roots.end());std::vector<double>lc={blo};for(double r:roots)if(r>lc.back()+eps)lc.push_back(r);lc.push_back(bhi);for(size_t j=0;j+1<lc.size();++j){double lo=lc[j],hi=lc[j+1];if(hi<=lo+eps)continue;double sample=.5*(lo+hi);auto win=*std::min_element(active.begin(),active.end(),[&](const Ref&a,const Ref&b){return segment_w_abs(passes[a.p].segments[a.s],sample)<segment_w_abs(passes[b.p].segments[b.s],sample);});append_env(out,win.p,win.s,passes[win.p].segments[win.s],lo,hi,eps);}}
    if (out.empty() || out.front().abs0 > eps || out.back().abs1 < total_L - eps) {
        fail(AME_REVERSE_TOPOLOGY_FAILURE, "candidate envelope incomplete");
    }
    for (size_t i = 1; i < out.size(); ++i) {
        if (out[i].abs0 > out[i - 1].abs1 + eps) {
            fail(AME_REVERSE_TOPOLOGY_FAILURE, "candidate envelope gap");
        }
    }
    return out;
}

static bool cap_releases(const TraversalPiece&p,double k){return p.sigma*k<0;}
static std::vector<InternalCapAnchor> collect_anchors(const std::vector<double>&raw,double initial_k,double margin){int n=(int)raw.size()/2;double anchor_g2=std::max({INTERNAL_CAP_G2_MARGIN,64*margin,512*ulp(M2)}),num=std::sqrt(std::max(0.0,M2-anchor_g2));std::vector<InternalCapAnchor>a;double station=0,k=initial_k;for(int i=0;i<n;++i){double L=raw[2*i],sigma=raw[2*i+1];station+=L;k=std::fma(sigma,L,k);int knot=i+1;if(knot==n)break;double right=raw[2*knot+1];bool rb=sigma*k>0,rf=right*k<0;if(!(rb||rf)||k==0)continue;double cap=num/std::fabs(k);if(cap>=W_EQ)continue;a.push_back({knot,station,k,cap,-cap/k});}return a;}
static std::vector<ScalarPass> compile_anchor_passes(const std::vector<TraversalPiece>&f,const std::vector<TraversalPiece>&b,const InternalCapAnchor&a,const ame_reverse_options&o){int n=(int)f.size(),j=a.knot_index;std::vector<ScalarPass>out;if(j<n){std::vector<TraversalPiece>fp(f.begin()+j,f.end());if(!fp.empty()&&cap_releases(fp[0],a.signed_k))out.push_back(build_scalar_pass(fp,a.cap_w,a.signed_k,PassKind::Forward,o,j,a.initial_w_dk));}int off=n-j;if(off<(int)b.size()){std::vector<TraversalPiece>bp(b.begin()+off,b.end());if(!bp.empty()&&cap_releases(bp[0],a.signed_k))out.push_back(build_scalar_pass(bp,a.cap_w,a.signed_k,PassKind::Backward,o,j,a.initial_w_dk));}return out;}
static std::vector<AnchorWitness> coverage_witnesses(std::vector<ScalarPass>&passes,double total,double eps=1e-11){std::vector<std::pair<double,double>>iv;for(auto&p:passes)for(auto&r:p.segments){auto[lo,hi]=interval(r);lo=std::max(0.0,lo);hi=std::min(total,hi);if(hi>lo+eps)iv.push_back({lo,hi});}std::sort(iv.begin(),iv.end());if(iv.empty())return{{0,total,0,0}};std::vector<std::pair<double,double>>m;for(auto x:iv){if(m.empty()||x.first>m.back().second+eps)m.push_back(x);else if(x.second>m.back().second)m.back().second=x.second;}std::vector<AnchorWitness>w;double cur=0;for(auto x:m){if(x.first>cur+eps)w.push_back({cur,x.first,0,0});cur=std::max(cur,x.second);}if(cur<total-eps)w.push_back({cur,total,0,0});return w;}
static std::vector<AnchorWitness> continuity_witnesses(std::vector<EnvelopePiece>&env,std::vector<ScalarPass>&passes,double total,double atol=2e-9,double rtol=2e-10){(void)total;std::vector<AnchorWitness>w;for(size_t i=0;i+1<env.size();++i){auto&l=env[i];auto&r=env[i+1];if(r.abs0>l.abs1+1e-11*std::max(1.0,total)){w.push_back({l.abs1,r.abs0,0,0});continue;}double station=.5*(l.abs1+r.abs0);auto&lr=passes[l.pass_index].segments[l.source_index];auto&rr=passes[r.pass_index].segments[r.source_index];double wl=segment_w_abs(lr,l.abs1),wr=segment_w_abs(rr,r.abs0),jump=std::fabs(wl-wr),tol=atol+rtol*std::max({1.0,std::fabs(wl),std::fabs(wr)});if(jump>tol)w.push_back({station,station,1,jump});}return w;}
static void require_envelope_boundaries(std::vector<EnvelopePiece>&env,std::vector<ScalarPass>&passes,double total,double start_w,double end_w_max){if(env.empty())fail(AME_REVERSE_TOPOLOGY_FAILURE,"empty envelope");auto&f=env.front();auto&l=env.back();double a=segment_w_abs(passes[f.pass_index].segments[f.source_index],0),b=segment_w_abs(passes[l.pass_index].segments[l.source_index],total);double st=2e-9+2e-10*std::max({1.0,std::fabs(a),std::fabs(start_w)});if(std::fabs(a-start_w)>st)fail(AME_REVERSE_TOPOLOGY_FAILURE,"envelope fixed start speed mismatch");double et=2e-9+2e-10*std::max({1.0,std::fabs(b),std::fabs(end_w_max)});if(b>end_w_max+et)fail(AME_REVERSE_TOPOLOGY_FAILURE,"envelope terminal speed cap exceeded");}
static double anchor_distance(const InternalCapAnchor&a,const AnchorWitness&w){if(w.abs0<=a.station&&a.station<=w.abs1)return 0;return std::min(std::fabs(a.station-w.abs0),std::fabs(a.station-w.abs1));}
static std::vector<int> select_anchor_batch(const std::vector<AnchorWitness>&w,const std::vector<InternalCapAnchor>&inactive){std::vector<int>sel;for(const auto&wi:w){int best=-1;for(size_t i=0;i<inactive.size();++i)if(wi.abs0<=inactive[i].station&&inactive[i].station<=wi.abs1){if(best<0||std::pair<double,int>{inactive[i].cap_w,inactive[i].knot_index}<std::pair<double,int>{inactive[best].cap_w,inactive[best].knot_index})best=(int)i;}if(best<0&&!inactive.empty()){best=0;for(size_t i=1;i<inactive.size();++i){auto ka=std::make_tuple(anchor_distance(inactive[i],wi),inactive[i].cap_w,inactive[i].knot_index);auto kb=std::make_tuple(anchor_distance(inactive[best],wi),inactive[best].cap_w,inactive[best].knot_index);if(ka<kb)best=(int)i;}}if(best>=0&&std::find(sel.begin(),sel.end(),inactive[best].knot_index)==sel.end())sel.push_back(inactive[best].knot_index);}return sel;}

static void aggregate_scalar_stats(const ScalarBuildImpl&b,uint64_t&calls,uint64_t&steps);

static void build_internal_candidates(ScalarBuildImpl&build,double initial_k){double total=total_length(build.raw);auto anchors=collect_anchors(build.raw,initial_k,build.options.domain_margin);build.anchor_stats.possible_anchors=anchors.size();std::vector<int>active;
    while(true){auto coverage=coverage_witnesses(build.passes,total);build.anchor_stats.coverage_witnesses+=coverage.size();std::vector<AnchorWitness>w;
        if(!coverage.empty())w=coverage;else{auto env=merge_candidate_envelope(build.passes,total);auto cont=continuity_witnesses(env,build.passes,total);build.anchor_stats.continuity_witnesses+=cont.size();if(cont.empty()){require_envelope_boundaries(env,build.passes,total,build.passes[0].segments[0].w0,build.passes[1].segments[0].w0);build.envelope=std::move(env);return;}w=std::move(cont);}
        std::vector<InternalCapAnchor>inactive;for(auto&a:anchors)if(std::find(active.begin(),active.end(),a.knot_index)==active.end())inactive.push_back(a);if(inactive.empty())fail(AME_REVERSE_TOPOLOGY_FAILURE,"internal cap anchors exhausted");auto ids=select_anchor_batch(w,inactive);if(ids.empty())fail(AME_REVERSE_TOPOLOGY_FAILURE,"anchor selection no progress");++build.anchor_stats.rounds;if(build.anchor_stats.rounds>anchors.size())fail(AME_REVERSE_TOPOLOGY_FAILURE,"anchor rounds exceeded catalog");size_t inserted=0;for(int id:ids){if(std::find(active.begin(),active.end(),id)!=active.end())continue;auto it=std::find_if(anchors.begin(),anchors.end(),[&](const auto&a){return a.knot_index==id;});if(it==anchors.end())continue;active.push_back(id);build.anchor_stats.inserted.push_back(id);auto pp=compile_anchor_passes(build.forward_pieces,build.backward_pieces,*it,build.options);for(auto&p:pp)build.passes.push_back(std::move(p));++inserted;}if(!inserted)fail(AME_REVERSE_TOPOLOGY_FAILURE,"anchor insertion no progress");}
}

static std::unique_ptr<ScalarBuildImpl> scalar_build_create_impl(const double*rawp,size_t raw_count,const ame_reverse_options&opt){
    auto b=std::make_unique<ScalarBuildImpl>();
    struct MetricsScope {
        BuildWorkMetrics *previous;
        explicit MetricsScope(BuildWorkMetrics *m): previous(ACTIVE_BUILD_METRICS){ACTIVE_BUILD_METRICS=m;}
        ~MetricsScope(){ACTIVE_BUILD_METRICS=previous;}
    } scope(&b->build_work);
    b->raw=validate_raw(rawp,raw_count);b->options=opt;double init_w=opt.has_init_w?opt.init_w:std::max(1e-3,.05*W_EQ);double terminal_w_max=opt.has_terminal_w_max?opt.terminal_w_max:init_w;double fk=opt.initial_k,bk=opt.has_backward_init_k?opt.backward_init_k:geometry_endpoint_k(b->raw,fk);b->forward_pieces=forward_traversal(b->raw);b->backward_pieces=backward_traversal(b->raw);b->passes.reserve(2+b->raw.size());b->passes.push_back(build_scalar_pass(b->forward_pieces,init_w,fk,PassKind::Forward,opt));b->passes.push_back(build_scalar_pass(b->backward_pieces,terminal_w_max,bk,PassKind::Backward,opt,(int)b->raw.size()/2,0.0));if(b->passes[0].segments.empty()||b->passes[1].segments.empty())fail(AME_REVERSE_TOPOLOGY_FAILURE,"endpoint pass emitted no segment");build_internal_candidates(*b,fk);
    uint64_t retained_calls=0,retained_steps=0;aggregate_scalar_stats(*b,retained_calls,retained_steps);
    b->build_cflow_calls=b->build_work.discarded_cflow_calls+retained_calls;
    b->build_cflow_steps=b->build_work.discarded_cflow_steps+retained_steps;
    return b;
}

static double scalar_time_value(ScalarBuildImpl&b){double value=0;for(auto&e:b.envelope){if(e.pass_index<0||e.pass_index>=(int)b.passes.size())fail(AME_REVERSE_TOPOLOGY_FAILURE,"bad envelope pass");auto&p=b.passes[e.pass_index];if(e.source_index<0||e.source_index>=(int)p.segments.size())fail(AME_REVERSE_TOPOLOGY_FAILURE,"bad envelope source");auto&s=p.segments[e.source_index];double lo=std::min(e.local0,e.local1),hi=std::max(e.local0,e.local1);if(hi-lo<=1e-14*std::max({1.0,std::fabs(lo),std::fabs(hi)}))continue;double t0=0,t1=0;auto a=ame_segment_time(s.seg.get(),lo,&t0),c=ame_segment_time(s.seg.get(),hi,&t1);if(a!=AME_SEGMENT_OK||c!=AME_SEGMENT_OK)fail(AME_REVERSE_NUMERICAL_FAILURE,"scalar time segment");double v=t1-t0;if(!isfin(v)||v<-64*ulp(std::max(1.0,std::fabs(v))))fail(AME_REVERSE_NUMERICAL_FAILURE,"scalar time invalid");value+=std::max(0.0,v);}return value;}

static void aggregate_scalar_stats(const ScalarBuildImpl&b,uint64_t&calls,uint64_t&steps){calls=steps=0;for(auto&p:b.passes)for(auto&s:p.segments){ame_segment_cache_stats cs{};if(ame_segment_cache_stats_get(s.seg.get(),&cs)==AME_SEGMENT_OK){calls+=cs.cflow_calls;steps+=cs.local_steps;}}}

static void add_geometry_endpoint_adj(const std::vector<double>&raw,std::vector<double>&g,double a){if(a==0)return;for(size_t i=0;i<raw.size()/2;++i){g[2*i]+=a*raw[2*i+1];g[2*i+1]+=a*raw[2*i];}}
static void add_geometry_knot_adj(const std::vector<double>&raw,std::vector<double>&g,int knot,double a){if(a==0)return;if(knot<0||knot>(int)raw.size()/2)fail(AME_REVERSE_NUMERICAL_FAILURE,"bad knot adjoint");for(int i=0;i<knot;++i){g[2*i]+=a*raw[2*i+1];g[2*i+1]+=a*raw[2*i];}}

static std::vector<int> time_replay_limits(const ScalarBuildImpl&b){std::vector<int>lim(b.passes.size(),0);for(auto&e:b.envelope){if(e.pass_index<0||e.pass_index>=(int)lim.size())fail(AME_REVERSE_TOPOLOGY_FAILURE,"bad envelope pass index");lim[e.pass_index]=std::max(lim[e.pass_index],e.source_index+1);}return lim;}

static RevPass promote_pass(ScalarPass &sp,int n_pieces,int segment_limit,bool reverse_eta,bool validate_domain,double domain_margin,int domain_scan){RevPass rp;rp.scalar_pass=&sp;rp.n_pieces=n_pieces;int limit=segment_limit<0?(int)sp.segments.size():segment_limit;if(limit<0||limit>(int)sp.segments.size())fail(AME_REVERSE_INVALID_ARGUMENT,"bad reverse segment limit");rp.segments.reserve(limit);for(int i=0;i<limit;++i){auto&s=sp.segments[i];bool eta=reverse_eta&&s.mode==AME_SEGMENT_GRIP&&s.sigma!=0;SegmentPtr seg=compile_segment_native(s.mode,s.L_used,s.sigma,s.w0,s.k0,true,s.boundary_start,eta,eta,s.w1);
        if(validate_domain){DomainClip clip{};if(first_domain_clip(seg.get(),s.L_used,domain_margin,domain_scan,clip))fail(AME_REVERSE_DOMAIN,"reverse replay domain");}
        ame_segment_state_jac sj{};auto st=ame_segment_w_and_jac(seg.get(),s.L_used,&sj);if(st!=AME_SEGMENT_OK)fail(AME_REVERSE_NUMERICAL_FAILURE,"reverse endpoint jac");double k1=std::fma(s.sigma,s.L_used,s.k0);if(std::fabs(sj.w-s.w1)>1e-8||std::fabs(k1-s.k1)>1e-8)fail(AME_REVERSE_NUMERICAL_FAILURE,"reverse replay endpoint mismatch");RevSegment rn;rn.scalar=&s;rn.seg=std::move(seg);rn.endpoint_state=sj;rn.has_endpoint_state=true;if(s.event!=EventKind::PieceEnd){rn.F_cols=local_event_columns(rn.seg.get(),s.L_used,s.event,&sj);rn.F_L=rn.F_cols[0];rn.has_event_cols=true;if(!isfin(rn.F_L)||std::fabs(rn.F_L)<=1e-14)fail(AME_REVERSE_NUMERICAL_FAILURE,"singular event root in promotion");for(double x:rn.F_cols)if(!isfin(x))fail(AME_REVERSE_NUMERICAL_FAILURE,"bad event columns");}rp.segments.push_back(std::move(rn));}return rp;}

static std::unique_ptr<ReverseBuildImpl> promote_impl(ScalarBuildImpl&scalar,bool time_only){auto r=std::make_unique<ReverseBuildImpl>();r->scalar=&scalar;r->time_only=time_only;int n=(int)scalar.raw.size()/2;std::vector<int>lim=time_only?time_replay_limits(scalar):std::vector<int>(scalar.passes.size(),-1);r->passes.reserve(scalar.passes.size());for(size_t i=0;i<scalar.passes.size();++i)r->passes.push_back(promote_pass(scalar.passes[i],n,time_only?lim[i]:-1,time_only,scalar.options.validate_replay_domain,scalar.options.domain_margin,scalar.options.domain_scan));return r;}

static std::pair<std::vector<double>,std::pair<double,double>> reverse_pass(RevPass&rp,double seed_w,double seed_k,const std::vector<SegmentExtraAdj>*extras_ptr=nullptr){RawGradientAccumulator acc(rp.n_pieces);std::vector<SegmentExtraAdj>zero;if(!extras_ptr){zero.resize(rp.segments.size());extras_ptr=&zero;}const auto&extras=*extras_ptr;if(extras.size()!=rp.segments.size())fail(AME_REVERSE_INVALID_ARGUMENT,"reverse extras size");double aw=seed_w,ak=seed_k;int current=-999999;double adj_offset_after=0;for(int idx=(int)rp.segments.size()-1;idx>=0;--idx){auto&node=rp.segments[idx];auto&s=*node.scalar;auto&e=extras[idx];if(current!=s.traversal_index){current=s.traversal_index;adj_offset_after=0;}double a_w0=e.aw0,a_k0=e.ak0,a_sigma=e.asigma;if(e.aabs0!=0)acc.add_abs0(s,e.aabs0);double adj_offset_before=s.direction*e.aabs0;if(!node.has_endpoint_state)fail(AME_REVERSE_NUMERICAL_FAILURE,"reverse endpoint state cache missing");const auto&sj=node.endpoint_state;double a0=std::fma(aw,sj.jac[0],ak*sj.jac[4]),a1=std::fma(aw,sj.jac[1],ak*sj.jac[5]),a2=std::fma(aw,sj.jac[2],ak*sj.jac[6]),a3=std::fma(aw,sj.jac[3],ak*sj.jac[7]);double adj_L=e.aL+a0+adj_offset_after;adj_offset_before+=adj_offset_after;if(s.event==EventKind::PieceEnd){acc.add_length(s.piece_index,adj_L);adj_offset_before-=adj_L;a_sigma+=a1;a_w0+=a2;a_k0+=a3;}else{if(!node.has_event_cols)fail(AME_REVERSE_NUMERICAL_FAILURE,"event missing columns");double scale=adj_L/node.F_L;a_sigma+=a1-scale*node.F_cols[1];a_w0+=a2-scale*node.F_cols[2];a_k0+=a3-scale*node.F_cols[3];}acc.add_traversal_sigma(s,a_sigma);aw=a_w0;ak=a_k0;adj_offset_after=adj_offset_before;}return{acc.finalize(),{aw,ak}};}

static std::pair<double,std::array<double,4>> prefix_time_jac(RevSegment&node,double x){auto&s=*node.scalar;double L=s.L_used;if(x<0&&x>-1e-12)x=0;if(x>L&&x<L+1e-12)x=L;if(x<0||x>L+1e-10)fail(AME_REVERSE_INVALID_ARGUMENT,"prefix outside segment");if(std::fabs(x)<=1e-14){if(!(s.w0>0&&isfin(s.w0)))fail(AME_REVERSE_NUMERICAL_FAILURE,"zero prefix w0");return{0,{1/std::sqrt(s.w0),0,0,0}};}ame_segment_time_jac tj{};auto st=ame_segment_time_and_jac(node.seg.get(),x,&tj);if(st!=AME_SEGMENT_OK)fail(AME_REVERSE_NUMERICAL_FAILURE,"prefix time jac");return{tj.time,{tj.jac[0],tj.jac[1],tj.jac[2],tj.jac[3]}};}

static std::tuple<double,double,double> seed_envelope_time(const EnvelopePiece&ep,RevSegment&node,SegmentExtraAdj&extra){auto&s=*node.scalar;double l0=ep.local0,l1=ep.local1;if(l0<0&&l0>-1e-12)l0=0;if(l0>s.L_used&&l0<s.L_used+1e-12)l0=s.L_used;if(l1<0&&l1>-1e-12)l1=0;if(l1>s.L_used&&l1<s.L_used+1e-12)l1=s.L_used;bool loep=l0<=l1;double lo=loep?l0:l1,hi=loep?l1:l0;if(hi-lo<=1e-14)return{0,0,0};auto[Tlo,Jlo]=prefix_time_jac(node,lo);auto[Thi,Jhi]=prefix_time_jac(node,hi);extra.asigma+=Jhi[1]-Jlo[1];extra.aw0+=Jhi[2]-Jlo[2];extra.ak0+=Jhi[3]-Jlo[3];double alo=-Jlo[0],ahi=Jhi[0],ae0=loep?alo:ahi,ae1=loep?ahi:alo;double aabs0=s.direction*ae0,aabs1=s.direction*ae1;extra.aabs0+=-s.direction*(ae0+ae1);return{Thi-Tlo,aabs0,aabs1};}
static int endpoint_kind(double x,double total){double tol=1e-10*std::max(1.0,total);if(std::fabs(x)<=tol)return 0;if(std::fabs(x-total)<=tol)return 2;return 1;}

static std::pair<double,std::vector<double>> reverse_time(ReverseBuildImpl&r){auto&b=*r.scalar;int n=(int)b.raw.size()/2;double total=total_length(b.raw),value=0;RawGradientAccumulator boundary(n);std::vector<std::vector<SegmentExtraAdj>>extras(r.passes.size());for(size_t p=0;p<r.passes.size();++p)extras[p].resize(r.passes[p].segments.size());for(auto&ep:b.envelope){if(ep.pass_index<0||ep.pass_index>=(int)r.passes.size())fail(AME_REVERSE_TOPOLOGY_FAILURE,"bad envelope pass in reverse");auto&rp=r.passes[ep.pass_index];if(ep.source_index<0||ep.source_index>=(int)rp.segments.size())fail(AME_REVERSE_TOPOLOGY_FAILURE,"time promotion omitted envelope source");auto[T,a0,a1]=seed_envelope_time(ep,rp.segments[ep.source_index],extras[ep.pass_index][ep.source_index]);value+=T;if(endpoint_kind(ep.abs0,total)==2)boundary.add_global_end(a0);if(endpoint_kind(ep.abs1,total)==2)boundary.add_global_end(a1);}auto grad=boundary.finalize();for(size_t p=0;p<r.passes.size();++p){auto[g,init]=reverse_pass(r.passes[p],0,0,&extras[p]);for(size_t i=0;i<grad.size();++i)grad[i]+=g[i];int knot=b.passes[p].initial_knot_index;if(knot>=0)add_geometry_knot_adj(b.raw,grad,knot,init.second+init.first*b.passes[p].initial_w_dk);}return{value,std::move(grad)};}

static std::pair<std::vector<double>,std::vector<double>> final_state_rows(ReverseBuildImpl&r,PassKind which){if(r.time_only)fail(AME_REVERSE_INVALID_ARGUMENT,"full promotion required for final rows");auto&b=*r.scalar;size_t pidx=which==PassKind::Forward?0:1;if(pidx>=r.passes.size())fail(AME_REVERSE_TOPOLOGY_FAILURE,"missing endpoint pass");auto&rp=r.passes[pidx];if(rp.segments.size()!=rp.scalar_pass->segments.size())fail(AME_REVERSE_TOPOLOGY_FAILURE,"incomplete full promotion");std::vector<std::vector<double>>rows;for(auto seed:{std::pair<double,double>{1,0},{0,1}}){auto[g,init]=reverse_pass(rp,seed.first,seed.second);if(which==PassKind::Backward)add_geometry_endpoint_adj(b.raw,g,init.second);rows.push_back(std::move(g));}return{std::move(rows[0]),std::move(rows[1])};}

static void aggregate_reverse_stats(const ReverseBuildImpl&r,uint64_t&calls,uint64_t&steps,size_t&grips){calls=steps=0;grips=0;for(auto&p:r.passes)for(auto&s:p.segments){if(s.scalar->mode==AME_SEGMENT_GRIP)++grips;ame_segment_cache_stats cs{};if(ame_segment_cache_stats_get(s.seg.get(),&cs)==AME_SEGMENT_OK){calls+=cs.cflow_calls;steps+=cs.local_steps;}}}

} // namespace

struct ame_scalar_build { std::unique_ptr<ScalarBuildImpl> p; };
struct ame_reverse_build { std::unique_ptr<ReverseBuildImpl> p; ame_scalar_build *owner = nullptr; };

static ame_reverse_status capture_error(const char *where) noexcept { try { throw; } catch(const ReverseError&e){LAST_ERROR=std::string(where)+": "+e.what();return e.status;}catch(const std::bad_alloc&e){LAST_ERROR=std::string(where)+": allocation failure";return AME_REVERSE_ALLOCATION_FAILURE;}catch(const std::exception&e){LAST_ERROR=std::string(where)+": "+e.what();return AME_REVERSE_NUMERICAL_FAILURE;}catch(...){LAST_ERROR=std::string(where)+": unknown failure";return AME_REVERSE_NUMERICAL_FAILURE;} }

extern "C" {
ame_reverse_options ame_reverse_default_options(void){ame_reverse_options o{};o.has_init_w=0;o.init_w=0;o.has_terminal_w_max=0;o.terminal_w_max=0;o.initial_k=0;o.n_scan=256;o.domain_scan=256;o.domain_margin=FRICTION_DOMAIN_MARGIN;o.fused_grip_discovery=1;o.validate_replay_domain=0;return o;}

ame_reverse_status ame_scalar_build_create(const double*raw,size_t n,const ame_reverse_options*opts,ame_scalar_build**out){if(!out)return AME_REVERSE_INVALID_ARGUMENT;*out=nullptr;try{ame_reverse_options o=opts?*opts:ame_reverse_default_options();if(o.n_scan<=0||o.domain_scan<=0||!isfin(o.domain_margin)||o.domain_margin<=0)fail(AME_REVERSE_INVALID_ARGUMENT,"invalid reverse options");if(o.has_terminal_w_max&&(!isfin(o.terminal_w_max)||o.terminal_w_max<=0||o.terminal_w_max>W_EQ))fail(AME_REVERSE_INVALID_ARGUMENT,"invalid terminal_w_max");auto h=std::make_unique<ame_scalar_build>();h->p=scalar_build_create_impl(raw,n,o);*out=h.release();LAST_ERROR.clear();return AME_REVERSE_OK;}catch(...){return capture_error("scalar_build_create");}}
void ame_scalar_build_destroy(ame_scalar_build*b){delete b;}
size_t ame_scalar_build_pass_count(const ame_scalar_build*b){return b&&b->p?b->p->passes.size():0;}
size_t ame_scalar_build_segment_count(const ame_scalar_build*b){if(!b||!b->p)return 0;size_t n=0;for(auto&p:b->p->passes)n+=p.segments.size();return n;}
size_t ame_scalar_build_envelope_count(const ame_scalar_build*b){return b&&b->p?b->p->envelope.size():0;}
size_t ame_scalar_build_inserted_anchor_count(const ame_scalar_build*b){return b&&b->p?b->p->anchor_stats.inserted.size():0;}
int ame_scalar_build_inserted_anchor_at(const ame_scalar_build*b,size_t i){if(!b||!b->p||i>=b->p->anchor_stats.inserted.size())return -1;return b->p->anchor_stats.inserted[i];}
ame_reverse_status ame_scalar_build_get_stats(const ame_scalar_build*b,ame_scalar_build_stats*out){if(!b||!b->p||!out)return AME_REVERSE_INVALID_ARGUMENT;try{std::memset(out,0,sizeof(*out));out->raw_values=b->p->raw.size();out->pieces=b->p->raw.size()/2;out->scalar_passes=b->p->passes.size();for(auto&p:b->p->passes)out->scalar_segments+=p.segments.size();out->envelope_pieces=b->p->envelope.size();out->possible_anchors=b->p->anchor_stats.possible_anchors;out->inserted_anchors=b->p->anchor_stats.inserted.size();out->anchor_rounds=b->p->anchor_stats.rounds;out->segment_compiles=b->p->build_work.segment_compiles;out->crossing_calls=b->p->build_work.crossing_calls;out->build_cflow_calls=b->p->build_cflow_calls;out->build_cflow_local_steps=b->p->build_cflow_steps;aggregate_scalar_stats(*b->p,out->scalar_cflow_calls,out->scalar_cflow_local_steps);return AME_REVERSE_OK;}catch(...){return capture_error("scalar_build_stats");}}
ame_reverse_status ame_scalar_build_segment_at(const ame_scalar_build*b,size_t flat,ame_scalar_segment_view*out){if(!b||!b->p||!out)return AME_REVERSE_INVALID_ARGUMENT;size_t k=0;for(size_t p=0;p<b->p->passes.size();++p)for(auto&s:b->p->passes[p].segments){if(k++==flat){out->pass_index=(int)p;out->pass_kind=public_pass(s.kind);out->mode=public_mode(s.mode);out->event=public_event(s.event);out->traversal_index=s.traversal_index;out->piece_index=s.piece_index;out->initial_knot_index=b->p->passes[p].initial_knot_index;out->boundary_start=s.boundary_start;out->offset0=s.offset0;out->L_used=s.L_used;out->sigma=s.sigma;out->abs0=s.abs0;out->abs1=s.abs1;out->direction=s.direction;out->w0=s.w0;out->k0=s.k0;out->w1=s.w1;out->k1=s.k1;return AME_REVERSE_OK;}}return AME_REVERSE_INVALID_ARGUMENT;}
ame_reverse_status ame_scalar_build_envelope_at(const ame_scalar_build*b,size_t i,ame_envelope_piece_view*out){if(!b||!b->p||!out||i>=b->p->envelope.size())return AME_REVERSE_INVALID_ARGUMENT;auto&e=b->p->envelope[i];out->source=public_pass(e.source);out->source_index=e.source_index;out->pass_index=e.pass_index;out->abs0=e.abs0;out->abs1=e.abs1;out->local0=e.local0;out->local1=e.local1;return AME_REVERSE_OK;}
ame_reverse_status ame_scalar_build_time_value(ame_scalar_build*b,double*out){if(!b||!b->p||!out)return AME_REVERSE_INVALID_ARGUMENT;try{*out=scalar_time_value(*b->p);LAST_ERROR.clear();return AME_REVERSE_OK;}catch(...){return capture_error("scalar_time");}}
ame_reverse_status ame_scalar_build_time_value_gradient(ame_scalar_build*b,double*out_v,double*out_g,size_t n){if(!b||!b->p||!out_v||!out_g)return AME_REVERSE_INVALID_ARGUMENT;try{if(n!=b->p->raw.size())fail(AME_REVERSE_INVALID_ARGUMENT,"gradient size mismatch");auto r=promote_impl(*b->p,true);auto[v,g]=reverse_time(*r);*out_v=v;std::copy(g.begin(),g.end(),out_g);LAST_ERROR.clear();return AME_REVERSE_OK;}catch(...){return capture_error("scalar_time_value_gradient");}}

static ame_reverse_status promote_common(ame_scalar_build*s,ame_reverse_build**out,bool time_only){if(!s||!s->p||!out)return AME_REVERSE_INVALID_ARGUMENT;*out=nullptr;try{auto r=std::make_unique<ame_reverse_build>();r->p=promote_impl(*s->p,time_only);r->owner=s;*out=r.release();LAST_ERROR.clear();return AME_REVERSE_OK;}catch(...){return capture_error(time_only?"promote_time":"promote_full");}}
ame_reverse_status ame_reverse_promote_time(ame_scalar_build*s,ame_reverse_build**out){return promote_common(s,out,true);}
ame_reverse_status ame_reverse_promote_full(ame_scalar_build*s,ame_reverse_build**out){return promote_common(s,out,false);}
void ame_reverse_build_destroy(ame_reverse_build*b){delete b;}
ame_reverse_status ame_reverse_build_get_stats(const ame_reverse_build*b,ame_reverse_stats*out){if(!b||!b->p||!out)return AME_REVERSE_INVALID_ARGUMENT;std::memset(out,0,sizeof(*out));out->promoted_passes=b->p->passes.size();for(auto&p:b->p->passes)out->promoted_segments+=p.segments.size();aggregate_reverse_stats(*b->p,out->replay_cflow_calls,out->replay_cflow_local_steps,out->promoted_grip_segments);return AME_REVERSE_OK;}
ame_reverse_status ame_reverse_time_value_gradient(ame_reverse_build*b,double*out_value,double*out_grad,size_t n){if(!b||!b->p||!out_value||!out_grad)return AME_REVERSE_INVALID_ARGUMENT;try{if(n!=b->p->scalar->raw.size())fail(AME_REVERSE_INVALID_ARGUMENT,"gradient size mismatch");auto[v,g]=reverse_time(*b->p);*out_value=v;std::copy(g.begin(),g.end(),out_grad);LAST_ERROR.clear();return AME_REVERSE_OK;}catch(...){return capture_error("reverse_time");}}
ame_reverse_status ame_reverse_final_state_rows(ame_reverse_build*b,ame_reverse_pass_kind which,double*out_w,double*out_k,size_t n){if(!b||!b->p||!out_w||!out_k)return AME_REVERSE_INVALID_ARGUMENT;try{if(n!=b->p->scalar->raw.size())fail(AME_REVERSE_INVALID_ARGUMENT,"row size mismatch");PassKind k=which==AME_REVERSE_PASS_FORWARD?PassKind::Forward:which==AME_REVERSE_PASS_BACKWARD?PassKind::Backward:throw ReverseError(AME_REVERSE_INVALID_ARGUMENT,"bad pass kind");auto[a,c]=final_state_rows(*b->p,k);std::copy(a.begin(),a.end(),out_w);std::copy(c.begin(),c.end(),out_k);return AME_REVERSE_OK;}catch(...){return capture_error("final_state_rows");}}
ame_reverse_status ame_reverse_time_value_gradient_raw(const double*raw,size_t n,const ame_reverse_options*o,double*out_v,double*out_g,size_t gn){ame_scalar_build*s=nullptr;auto st=ame_scalar_build_create(raw,n,o,&s);if(st!=AME_REVERSE_OK)return st;ame_reverse_build*r=nullptr;st=ame_reverse_promote_time(s,&r);if(st==AME_REVERSE_OK)st=ame_reverse_time_value_gradient(r,out_v,out_g,gn);ame_reverse_build_destroy(r);ame_scalar_build_destroy(s);return st;}
const char*ame_reverse_last_error(void){return LAST_ERROR.c_str();}
const char*ame_reverse_status_name(ame_reverse_status s){switch(s){case AME_REVERSE_OK:return"ok";case AME_REVERSE_INVALID_ARGUMENT:return"invalid_argument";case AME_REVERSE_DOMAIN:return"domain";case AME_REVERSE_CONDITIONING:return"conditioning";case AME_REVERSE_NUMERICAL_FAILURE:return"numerical_failure";case AME_REVERSE_TOPOLOGY_FAILURE:return"topology_failure";case AME_REVERSE_ALLOCATION_FAILURE:return"allocation_failure";case AME_REVERSE_UNSUPPORTED:return"unsupported";default:return"unknown";}}
}
