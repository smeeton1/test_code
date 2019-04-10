program classical_ai
implicit none
real,dimension(:,:),allocatable  :: x,syn0,l1,l1_d,h2
real,dimension(:),allocatable    :: y,syn1,l2,l2_d,h
integer                          :: i,j,k
real                             :: r


allocate(x(4,3),y(4),syn0(3,4),syn1(4),l1(4,4),l1_d(4,4),l2(4),l2_d(4),h2(3,4),h(4))

do i=1,4
  do j=1,3
    syn0(j,i)=2*rand()-1
  
  enddo
  syn1(i)=2*rand()-1
enddo
x=reshape((/ 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1 /), shape(x))
y=reshape((/ 0, 0, 1, 1 /), shape(y))
write(*,*)x(1,:)
write(*,*)x(2,:)
write(*,*)x(3,:)
write(*,*)x(4,:)

write(*,*)y
do i=1,60000
  
  l1=matmul(x,syn0)
  
  do j=1,4
    do k=1,4
      l1(j,k)=1/(1+exp(-l1(j,k)))
    enddo
  enddo
  
  l2=matmul(l1,syn1)
  
  do j=1,4
    l2(j)=1/(1+exp(-l2(j)))
    l2_d(j)=(y(j)-l2(j))*(l2(j)*(1-l2(j)))
  enddo
  
  do j=1,4
    do k=1,4
      l1_d(j,k)=l2_d(j)*syn1(k)*(l1(j,k)*(1-l1(j,k)))
    enddo
  enddo  
  h=matmul(transpose(l1),l2_d)
  do j=1,4
    syn1(j)=syn1(j)+h(j)
  enddo
  h2=matmul(transpose(x),l1_d)
  do j=1,4
    do k=1,4
      syn0(j,k)=syn0(j,k)+h2(j,k)
    enddo
  enddo  
  
  

enddo


write(*,*)l2






end program