using ITensors

function check_str(a)
    try
        parse(Int64,a)
        true
    catch
        false
    end
end
 
fname = []
oname = []
verbous = false
 
for i=1:size(ARGS,1)
 if ARGS[i] == "-f"
   i=i+1
   push!(fname,ARGS[i])
 elseif ARGS[i] == "-h"
   println(" ")
 elseif ARGS[i] == "-o"
   i=i+1
   push!(oname,ARGS[i])
 elseif ARGS[i] == "-v"
   verbous = true
 end 
end 
  println(fname)
 
 if !isempty(oname)
    println("empty")
 end
 
 
 
 
 a=[]
 open(fname[1]) do f
   i=1
   while !eof(f)
    x=readline(f)
    append!(a,[split(x)])
#    println(x)
   end
 end
if check_str(a[1][1])
 # println(parse(Int,a[1][1]))
end
# println(a[3][3])
println(a)
b=[]
for i =1:size(a,1)
    c=[]
    for j =1:size(a[i],1)
       
       if check_str(a[i][j])
          append!(b,parse(Int64,a[i][j]))
       else
          append!(b,a[i][j])
       end
    end
    #append!(b,c)
end
if check_str(a[1][1])
    d=Index[]
    for i=1:parse(Int64,a[1][1])
        append!(d,[Index(4),Index(4)])
    end
end

l=Index(4,"l")
# reshape(b,(3,2))
# 
 println(d)
 println(d[3])
 d=reshape(d,(parse(Int64,a[1][1]),2))
 println(d)
 println(d[2,1])
# println(size(b,1))
# println(size(a,1))
