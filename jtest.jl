
for arg in ARGS
 println(arg)
end

x=1
y=x
for i=1:5
  global x=x+1
end
  
println(x)
println(y)

z=[[1,2],[2,2],[3,3]]
println(z)

