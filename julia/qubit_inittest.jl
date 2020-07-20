using ITensors      

function HGate(i::Index, j::Index)
  A = ITensor(i,j)

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
  A = ITensor(i,j)

  A[i(1),j(4)]=1
  A[i(2),j(3)]=1
  A[i(3),j(2)]=1
  A[i(4),j(1)]=1

  return A

end 

function CNotGate(i::Index, j::Index, k::Index, l::Index)
  A = ITensor(i,j,k,l)

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
  A = ITensor(i)

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
     if !isempty(Ham)
       if hasinds(Ham[end],d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2])&&(order(Ham[end])==2)
         Ham[end]=Ham[end]+HGate(d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2])
       else
         push!(Ham,HGate(d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2]) )
       end
     else
       push!(Ham,HGate(d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2]) )
     end
   end
   
   if a[i][1] == "X"
     if !isempty(Ham)
       if hasinds(Ham[end],d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2])&&(order(Ham[end])==2)
         Ham[end]=Ham[end]+XGate(d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2])
       else
         push!(Ham,XGate(d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2]) )
       end
     else
       push!(Ham,XGate(d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2]) )
     end
   end
   
   if a[i][1] == "CN"
     if !isempty(Ham)
       if hasinds(Ham[end],d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2],d[parse(Int64,a[i][3]),1],d[parse(Int64,a[i][3]),2])&&(order(Ham[end])==4)
         Ham[end]=Ham[end]+CNotGate(d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2],d[parse(Int64,a[i][3]),1],d[parse(Int64,a[i][3]),2])
       else
         push!(Ham,CNotGate(d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2],d[parse(Int64,a[i][3]),1],d[parse(Int64,a[i][3]),2]) )
       end
     else
       push!(Ham,CNotGate(d[parse(Int64,a[i][2]),1],d[parse(Int64,a[i][2]),2],d[parse(Int64,a[i][3]),1],d[parse(Int64,a[i][3]),2]) )
     end
   end
 end

 return Ham
end

function Ten_Add(Tens)
 fin=[]
 N = size(Tens,1)
 #println(N)
 re=[]
 #append!(fin,Tens[1])
 append!(re,0)
 for i=1:N
  # println(Tens[i])
   if !(i in re)
   push!(fin,Tens[i])
   ad = true
   j=i+1
   while ad
    #println(j)
    if j>N
      ad = false
    elseif hasinds(Tens[j],Tens[i])
     if order(Tens[i])==order(Tens[j])
        fin[i]=fin[i]+Tens[j]
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
Ham = Ten_Add(Ham)
println('5')
T= CNotGate(d[1,1],d[1,2],d[2,1],d[2,2])
println(T)
println("svd of T")
U,S,V =svd(T,(d[1,1],d[1,2]))
println("U=",U)
println("S=",S)
println("V=",V)
#println(hasinds(Ham[1],Ham[2]))
println(size(Ham,1))
println(Ham)
 
 
 
