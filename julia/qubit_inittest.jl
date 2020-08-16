using ITensors  
using Random




function HGate(i::Index, j::Index)
  A = ITensor(ComplexF64,i,j)

  A[i(1),j(1)]=1.0/2.0
  A[i(1),j(2)]=1.0/2.0
  A[i(1),j(3)]=1.0/2.0
  A[i(1),j(4)]=1.0/2.0
  A[i(2),j(1)]=1.0/2.0
  A[i(2),j(2)]=-1.0/2.0
  A[i(2),j(3)]=1.0/2.0
  A[i(2),j(4)]=-1.0/2.0
  A[i(3),j(1)]=1.0/2.0
  A[i(3),j(2)]=1.0/2.0
  A[i(3),j(3)]=-1.0/2.0
  A[i(3),j(4)]=-1.0/2.0
  A[i(4),j(1)]=1.0/2.0
  A[i(4),j(2)]=-1.0/2.0
  A[i(4),j(3)]=-1.0/2.0
  A[i(4),j(4)]=1.0/2.0
  
  return A

end 

function XGate(i::Index, j::Index)
  A = ITensor(ComplexF64,i,j)

  A[i(1),j(4)]=1
  A[i(2),j(3)]=1
  A[i(3),j(2)]=1
  A[i(4),j(1)]=1

  return A

end 

function CNotGate(i::Index, j::Index, k::Index, l::Index)
  A = ITensor(ComplexF64,i,j,k,l)

  A[i(1),j(1),k(1),l(1)]=1
  A[i(1),j(2),k(1),l(2)]=1
  A[i(1),j(3),k(1),l(3)]=1
  A[i(1),j(4),k(1),l(4)]=1
  A[i(2),j(1),k(2),l(2)]=1
  A[i(2),j(2),k(2),l(1)]=1
  A[i(2),j(3),k(2),l(4)]=1
  A[i(2),j(4),k(2),l(3)]=1
  A[i(3),j(1),k(3),l(3)]=1
  A[i(3),j(2),k(3),l(4)]=1
  A[i(3),j(3),k(3),l(1)]=1
  A[i(3),j(4),k(3),l(2)]=1
  A[i(4),j(1),k(4),l(4)]=1
  A[i(4),j(2),k(4),l(3)]=1
  A[i(4),j(3),k(4),l(2)]=1
  A[i(4),j(4),k(4),l(1)]=1

  return A

end 


function Init_st(i::Index,a)
  A = ITensor(ComplexF64,i)

  if isa(a,Array)
  A[i(1)]=parse(ComplexF64,a[1])
  if size(a,1)>=2
    A[i(2)]=parse(ComplexF64,a[2])
        if size(a,1)>=3
            A[i(3)]=parse(ComplexF64,a[3])
                if size(a,1)>=4
                    A[i(4)]=parse(ComplexF64,a[4])
                else
                    A[i(4)]= 0.0 + 0.0im
                end
        else
            A[i(3)]= 0.0 + 0.0im
            A[i(4)]= 0.0 + 0.0im
        end
  else
    A[i(2)]= 0.0 + 0.0im
    A[i(3)]= 0.0 + 0.0im
    A[i(4)]= 0.0 + 0.0im
  end
  else
   A[i(1)]=parse(ComplexF64,a)
   A[i(2)]= 0.0 + 0.0im
   A[i(3)]= 0.0 + 0.0im
   A[i(4)]= 0.0 + 0.0im
  end 
  return A
  
end


function Inital_State(N,d,a)
 Q=ITensor[]
 for i=1:N
 #println(d[i,2],a[1+i])
  push!(Q,Init_st(d[i,2],a[1+i]))
 end
 return Q
end

function Index_setup(a)

    d=Index[]
    for i=1:a
        append!(d,[Index(4),Index(4)])
    end
    d=reshape(d,(a,2))

end

function Tensor_Setup(N,d,a)
 Ham=[]
 for i=N+2:size(a,1)
   if a[i][1] == "H"
     
       push!(Ham,HGate(d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2]) )
     
   end
   
   if a[i][1] == "X"

       push!(Ham,XGate(d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2]) )
     
   end
   
   if a[i][1] == "CN"
    
       push!(Ham,CNotGate(d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2],d[parse(Int64,a[i][3]),1],d[parse(Int64,a[i][3]),2]) )
   end
 end

 return Ham
end

function Ten_Add(Tens)
 fin=[]
 N = size(Tens,1)
 println("Add N=",N)
 re=[]
 #append!(fin,Tens[1])
 append!(re,0)
 k=0
 for i=1:N
  # println(Tens[i])
   if !(i in re)
   push!(fin,Tens[i])

   k=k+1
   index=inds(fin[k])
   ad = true
   j=i+1
   while ad
    println("j=",j)
    if j>N
      ad = false
      
    elseif hasinds(Tens[j],index)
     if order(fin[k])==order(Tens[j])
        fin[k]=fin[k]+Tens[j]
        append!(re,j)
        j=j+1
     else
        ad = false
     end
    else
      j=j+1
    end
   
   end
   end
 
 end

 return fin
 
end

function Ten_split(T)
 fin=[]
 N = size(T,1)
 for i=1:N
  if order(T[i])>2
   index=inds(T[i])
   U,S,V =svd(T[i],(index[1],index[2])) 
   U=U*S
   push!(fin,U)
   while order(V)>3
    index=inds(V)
    n=size(index,1)
    U,S,V =svd(V,(index[1],index[2],index[n]))
    U=U*S
    push!(fin,U)
   end
   push!(fin,V)
  
  else
   push!(fin,T[i])
  end
 
 end

 return fin
 
end

function line_mps(Q,T)
 N=size(T,1)
 #println("N=" ,N)
 index=inds(Q)
 #println(index)
 for i=1:N
   #println(hasinds(T[i],index))
   if hasinds(T[i],index)
     Q=Q*T[i]
     #println(inds(T[i]))
  
   end
  
 end

 return Q

end

function Mtra(i::Index)
  A = ITensor(ComplexF64,i)

  A[i(1)]=1
  A[i(4)]=1

  return A

end 

function Par_Trac(T::ITensor, I::Index)
 if hasinds(T,I)

  index=inds(T)
  A=T
  for j=1:length(index)
    if index[j]!=I
     temp= Mtra(index[j])
     A= A*temp
    end

  end
  return A
 else
  println(I, "is not in T")
  return T
 end


end

function Contract_Lines(Q,T)
 N=size(Q,1)
 A=[]
 for i=1:N
   push!(A,line_mps(Q[i],T))
 end
 B=A[1]*A[2]
 for i=3:N
   B=B*A[i]
 end
 index=inds(B)
 #println(index)
 #println(length(index))
 for j=1:N
  A[j]=Par_Trac(B, index[j])
  #println("o= ",order(A[j]))
 end
 return A

end

function Contract_node(Q,T,d::Integer)
 N=size(Q,1)
 M=size(T,1)
 A=[]
 #do d contraction  along a given index


end

function Q_Meas(Q)
 i=0
 while i<30
  x=bitrand()
  if x[1]
   if rand() < abs(Q[1])
    return 1
   end
  else
   if rand() < abs(Q[4])
    return 0
   end
  end
  i=i+1
 end
 x =bitrand()
 if x[1]
    return 1
 else
    return 0
 end

end

function density_out(T)

  println(T[1],' ',T[2])
  println(T[3],' ',T[4])

end

function wave_out(T)
if sign(T[2]) == 0
 println(sqrt(T[1]),' ', sqrt(T[4]))
else
 println(sqrt(T[1]),' ', sign(T[2])*sqrt(T[4]))
end

end


 a=[]
 open("input") do f
   i=1
   while !eof(f)
    x=readline(f)
    append!(a,[split(x)])
#    println(x)
   end
 end
# println(a[1])
println('1')
d=Index_setup(parse(Int64,a[1][1])) 
#println(size(a[2],1))
#println(d)
println('2')
Q=Inital_State(parse(Int64,a[1][1]),d,a)

#println(Q)
# println(HGate(d[1,1],d[1,2]))
println('3')
Ham = Tensor_Setup(parse(Int64,a[1][1]),d,a)
println('4')
# println(size(Ham,1))
# println(Ham[4])
#Ham = Ten_Add(Ham)
println('5')
Ham = Ten_split(Ham)
println('6')
A=[]
A=Contract_Lines(Q,Ham)
println('7')

Tr=ITensor[]
push!(Tr,Mtra(d[1,1]))
push!(Tr,Mtra(d[2,2]))
push!(Tr,Mtra(d[3,2]))

Q[1]= line_mps(Q[1],Ham)
Q[2]= line_mps(Q[2],Ham)
Q[3]= line_mps(Q[3],Ham)


T1=Q[3]*Q[2]*Q[1]*Tr[2]*Tr[3]
T2=Q[1]*Q[2]*Q[3]
T3=Q[2]*Q[1]*Q[3]*Tr[1]*Tr[2]
T2=Par_Trac(T2, d[2,2])

# O1=Q[1]*Tr[1]
# O2=Q[2]*Tr[2]
# O3=Q[3]*Tr[3]
println('8')
# for j=1:10
#  println(Q_Meas(T1))
# end
density_out(T1)
wave_out(T1)
wave_out(T2)
# T1p=O3*O2*Q[1]
# T2p=O3*O1*Q[2]
# T3p=O2*O1*Q[3]
# if T1==T2
#  println("T")
# else
#  println("F")
# end
println('9')
println(T1)
#println(A[1])
println(T2)
#println(A[2])
println(T3)
#println(A[3])

# println(Q[1])
# println(Q[2])
# println(Q[3])
# println(norm(T1),norm(T2),norm(T3))
#println(hasinds(Ham[1],Ham[2]))
#println("Ham size =" ,size(Ham,1))
#println(Ham)
 
 
#     if a[i][1] == "Y"
#      if !isempty(Ham)
#        if hasinds(Ham[end],d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2])&&(order(Ham[end])==2)
#          Ham[end]=Ham[end]+YGate(d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2])
#        else
#          push!(Ham,YGate(d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2]) )
#        end
#      else
#        push!(Ham,YGate(d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2]) )
#      end
#    end
