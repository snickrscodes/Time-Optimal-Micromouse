#include "include/ame_crossing.h"
#include <chrono>
#include <cstdio>
#include <vector>

struct Case {const char*name;ame_segment_mode mode;ame_crossing_kind kind;double L,sigma,w0,k0;int safe;};
static const Case CASES[]={
 {"grip_motor_noevent",AME_SEGMENT_GRIP,AME_CROSSING_GRIP_MOTOR,.2,-.5,4.0,2.5,1},
 {"grip_motor_event",AME_SEGMENT_GRIP,AME_CROSSING_GRIP_MOTOR,.336637731908148,-1.4925091309288019,7.310500916772633,1.4666036774619606,1},
 {"grip_motor_domain",AME_SEGMENT_GRIP,AME_CROSSING_GRIP_MOTOR,.09596588186356513,-12.956218401401495,1.2959185485103393,-4.507559860908246,1},
 {"grip_brake_event",AME_SEGMENT_GRIP,AME_CROSSING_GRIP_BRAKE,.8,-3.0,11.772*.8/2.0,2.0,1},
 {"grip_brake_noevent",AME_SEGMENT_GRIP,AME_CROSSING_GRIP_BRAKE,.1,-.1,11.772*.55,1.0,1},
 {"motor_grip_event",AME_SEGMENT_MOTOR,AME_CROSSING_MOTOR_GRIP,.06505260165163275,-9.433050469559873,12.970782600365993,-.6723293209494665,0},
 {"brake_grip_event",AME_SEGMENT_BRAKE,AME_CROSSING_BRAKE_GRIP,.12396771062269442,-1.5576684883456533,.5472281286529663,-2.7830833372696495,0},
};
int main(){constexpr int N=1000,REPS=7;ame_crossing_options o=ame_crossing_default_options();o.n_scan=48;for(const auto&c:CASES){double best=1e300;for(int rep=0;rep<REPS;++rep){std::vector<ame_segment*>v;v.reserve(N);for(int i=0;i<N;++i){ame_segment_status st;ame_segment_options so=ame_segment_default_options();auto*s=ame_segment_compile(c.L,c.sigma,c.w0,c.k0,c.mode,0,&so,&st);if(!s)return 2;v.push_back(s);}auto t0=std::chrono::steady_clock::now();uint64_t sink=0;for(auto*s:v){ame_crossing_result r;auto st=ame_crossing_scan(s,c.L,c.kind,c.safe,&o,&r);if(st!=AME_CROSSING_OK)return 3;sink+=r.state_evaluations+(uint64_t)r.has_event;}auto t1=std::chrono::steady_clock::now();double ns=std::chrono::duration<double,std::nano>(t1-t0).count()/N;if(ns<best)best=ns;for(auto*s:v)ame_segment_destroy(s);if(sink==0xdeadbeef)std::puts("sink");}std::printf("%-24s %.3f ns\n",c.name,best);}return 0;}
