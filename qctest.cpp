
#include <iostream>
#include "itensor/all.h"
using namespace itensor;

/// compile line  g++-9 qctest.cpp -I/home/jrod/code/itensor/ /home/jrod/code/itensor/lib/libitensor.a -L/home/jrod/code/itensor -lpthread -L/usr/lib -lblas -llapack -std=c++17 -m64 -fPIC -o qctest


int main(){
  
  
  auto q1=Index(4,"q1");
//   auto w1=Index(4,"w1");
//   auto q2=Index(4,"q2");
//   auto w2=Index(4,"w2");
//   
/*  
  auto D1=ITensor(q1);
  auto D2=ITensor(q2);
  
  auto Cnot=ITensor(q1,w1,q2,w2);*/
  
//   D1.set(q1=1,1);
//   D2.set(q2=1,1);
//   
//   Cnot.set(q1=0,w1=0,q2=0,w2=0,1);
//   Cnot.set(q1=0,w1=1,q2=0,w2=1,1);
//   Cnot.set(q1=0,w1=2,q2=0,w2=2,1);
//   Cnot.set(q1=0,w1=3,q2=0,w2=3,1);
//   Cnot.set(q1=1,w1=0,q2=1,w2=1,1);
//   Cnot.set(q1=1,w1=1,q2=1,w2=0,1);
//   Cnot.set(q1=1,w1=2,q2=1,w2=3,1);
//   Cnot.set(q1=1,w1=3,q2=1,w2=2,1);
//   Cnot.set(q1=2,w1=0,q2=2,w2=2,1);
//   Cnot.set(q1=2,w1=1,q2=2,w2=3,1);
//   Cnot.set(q1=2,w1=2,q2=2,w2=0,1);
//   Cnot.set(q1=2,w1=3,q2=2,w2=1,1);
//   Cnot.set(q1=3,w1=0,q2=3,w2=3,1);
//   Cnot.set(q1=3,w1=1,q2=3,w2=2,1);
//   Cnot.set(q1=3,w1=2,q2=3,w2=1,1);
//   Cnot.set(q1=3,w1=3,q2=3,w2=0,1);
//   
//   auto O=D1*Cnot;
//   O=D2*O;
//   
//   PrintData(O);
  
  return 0;
}