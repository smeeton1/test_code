program small_t
implicit none
real,dimension(:,:),allocatable  :: syn0,x
real,dimension(:),allocatable    :: out,out_d,syn1,y
integer                          :: i,j,k,l,n_o,n_i,step
real                             :: r,r_d


step=10000
n_o=2
n_i=3
allocate(x(n_i,3),y(3),syn0(n_o,n_i),out(n_o),out_d(n_o),syn1(n_o))
x=reshape((/ 1, 0, 0 ,0,1,1,1,0,1/), shape(x))
y=reshape((/ 1,0,1/), shape(y))

do i=1,n_o
  do j=1,n_i
    syn0(j,i)=2*rand()-1
  
  enddo
  syn1(i)=2*rand()-1
enddo
! write(*,*)size(x)
! write(*,*)size(y)
! write(*,*)size(syn0,1)
! write(*,*)size(syn0,2)

do k=1,step


  do l=1,3
  !forward porp
    out=matmul(syn0,x(:,l))
  
    do i=1,n_o
      out(i)=1/(1+exp(-out(i)))
    enddo
  
    r=0
    do i=1,n_o
      r=r+(-syn1(i)*out(i))
    enddo
    r=1/(1+exp(r))
  !backwards prop

   r_d=(y(l)-r)*(r*(1-r))
   
   do i=1,n_o
     syn1(i)=syn1(i)+r_d*out(i)
   enddo
   
   do i=1,n_o
     out_d(i)=(r_d*syn1(i))*(out(i)*(1-out(i)))
   enddo
   
    do i=1,n_o
      do j=1,n_i
	syn0(i,j)=syn0(i,j)+x(j,l)*out_d(i)
      enddo
    enddo
    
  enddo

enddo

write(*,*) r


end program