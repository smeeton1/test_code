
#include <iostream>
#include "itensor/all.h"
using namespace itensor;
using namespace std;

/// compile line  g++-9 qctest.cpp -I/home/jrod/code/itensor/ /home/jrod/code/itensor/lib/libitensor.a -L/home/jrod/code/itensor -lpthread -L/usr/lib -lblas -llapack -std=c++17 -m64 -fPIC -o qctest


int main(){
  
  cout<<"0\n";
   
  auto q1=Index(4,"q1");
  auto w1=Index(4,"w1");
  auto q2=Index(4,"q2");
  auto w2=Index(4,"w2");
  
  
  auto D1=ITensor(w1);
  auto D2=ITensor(w2);
  
  auto Cnot=ITensor(q1,w1,q2,w2);
  
  auto H1=ITensor(q1,w1);
  
  auto H2=ITensor(q2,w2);
  
  auto X1=ITensor(q1,w1);
  
  auto XM1=ITensor(q1);
  
  auto XM2=ITensor(q2);
  
  auto ZM1=ITensor(w1);
  
  auto ZM2=ITensor(w2);
  
  
  cout<<"1\n";
  
  D1.set(w1=1,1);
  D2.set(w2=1,1);
  
  cout<<"2\n";
  
  Cnot.set(q1=1,w1=1,q2=1,w2=1,1);
  Cnot.set(q1=1,w1=2,q2=1,w2=2,1);
  Cnot.set(q1=1,w1=3,q2=1,w2=3,1);
  Cnot.set(q1=1,w1=4,q2=1,w2=4,1);
  Cnot.set(q1=2,w1=1,q2=2,w2=2,1);
  Cnot.set(q1=2,w1=2,q2=2,w2=1,1);
  Cnot.set(q1=2,w1=3,q2=2,w2=4,1);
  Cnot.set(q1=2,w1=4,q2=2,w2=3,1);
  Cnot.set(q1=3,w1=1,q2=3,w2=3,1);
  Cnot.set(q1=3,w1=2,q2=3,w2=4,1);
  Cnot.set(q1=3,w1=3,q2=3,w2=1,1);
  Cnot.set(q1=3,w1=4,q2=3,w2=2,1);
  Cnot.set(q1=4,w1=1,q2=4,w2=4,1);
  Cnot.set(q1=4,w1=2,q2=4,w2=3,1);
  Cnot.set(q1=4,w1=3,q2=4,w2=2,1);
  Cnot.set(q1=4,w1=4,q2=4,w2=1,1);
  
  cout<<"3\n";


  H1.set(q1=1,w1=2,1.0/2.0);
  H1.set(q1=1,w1=3,1.0/2.0);
  H1.set(q1=1,w1=4,1.0/2.0);
  H1.set(q1=2,w1=1,1.0/2.0);
  H1.set(q1=2,w1=2,-1.0/2.0);
  H1.set(q1=2,w1=3,1.0/2.0);
  H1.set(q1=2,w1=4,-1.0/2.0);
  H1.set(q1=3,w1=1,1.0/2.0);
  H1.set(q1=3,w1=2,1.0/2.0);
  H1.set(q1=3,w1=3,-1.0/2.0);
  H1.set(q1=3,w1=4,-1.0/2.0);
  H1.set(q1=4,w1=1,1.0/2.0);
  H1.set(q1=4,w1=2,-1.0/2.0);
  H1.set(q1=4,w1=3,-1.0/2.0);
  H1.set(q1=4,w1=4,1.0/2.0);
  H1.set(q1=1,w1=1,1.0/2.0);
  
  PrintData(H1);
  
  cout<<"4\n";
  

  H2.set(q2=1,w2=2,1.0/2.0);
  H2.set(q2=1,w2=3,1.0/2.0);
  H2.set(q2=1,w2=4,1.0/2.0);
  H2.set(q2=2,w2=1,1.0/2.0);
  H2.set(q2=2,w2=2,-1.0/2.0);
  H2.set(q2=2,w2=3,1.0/2.0);
  H2.set(q2=2,w2=4,-1.0/2.0);
  H2.set(q2=3,w2=1,1.0/2.0);
  H2.set(q2=3,w2=2,1.0/2.0);
  H2.set(q2=3,w2=3,-1.0/2.0);
  H2.set(q2=3,w2=4,-1.0/2.0);
  H2.set(q2=4,w2=1,1.0/2.0);
  H2.set(q2=4,w2=2,-1.0/2.0);
  H2.set(q2=4,w2=3,-1.0/2.0);
  H2.set(q2=4,w2=4,1.0/2.0);
  H2.set(q2=1,w2=1,1.0/2.0);
  cout<<"5\n";
  
  X1.set(q1=1,w1=4,1);
  X1.set(q1=2,w1=3,1);
  X1.set(q1=3,w1=2,1);
  X1.set(q1=4,w1=1,1);
  
  XM1.set(q1=2,1);
  XM1.set(q1=3,1);
  
  XM2.set(q2=2,1);
  XM2.set(q2=3,1);
  
  ZM1.set(w1=1,1);
  ZM1.set(w1=4,-1);
  
  ZM2.set(w2=1,1);
  ZM2.set(w2=4,-1);
  
  cout<<"6\n";
  
  //auto ho=H1+X1;
  auto O=D1*D2*(H1+X1)*Cnot*X1*XM1*XM2;
//   O=D2*O;
  
  PrintData(O);
  
  return 0;
}