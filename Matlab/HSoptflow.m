function [us,vs] = HSoptflow(Xrgb,n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Gregory Power gregory.power@wpafb.af.mil
% This MATLAB code shows a Motion Estimation map created by
% using a Horn and Schunck motion estimation technique on two
% consecutive frames.  Input requires.
%     Xrgb(h,w,d,N) where X is a frame sequence of a certain
%                height(h), width (w), depth (d=3 for color), 
%                and number of frames (N). 
%     n= is the starting frame number which is less than N 
%     V= the output variable which is a 2D velocity array
%
% Sample Call: V=HSoptflow(X,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[h,w,d,N]=size(Xrgb);
if n>N-1
   error(1,'requested file greater than frame number required');
end; 
%get two image frames
if d==1
    Xn=double(Xrgb(:,:,1,n));
    Xnp1=double(Xrgb(:,:,1,n+1));
elseif d==3
    Xn=double(Xrgb(:,:,1,n)*0.299+Xrgb(:,:,2,n)*0.587+Xrgb(:,:,3,n)*0.114);
    Xnp1=double(Xrgb(:,:,1,n+1)*0.299+Xrgb(:,:,2,n+1)*0.587+Xrgb(:,:,3,n+1)*0.114);
else
    error(2,'not an RGB or Monochrome image file');
end;

%get image size and adjust for border
size(Xn);
hm5=h-5; wm5=w-5;
z=zeros(h,w); v1=z; v2=z;

%initialize
dsx2=v1; dsx1=v1; dst=v1;
alpha2=625;
imax=20;

%Calculate gradients
dst(5:hm5,5:wm5) = ( Xnp1(6:hm5+1,6:wm5+1)-Xn(6:hm5+1,6:wm5+1) + Xnp1(6:hm5+1,5:wm5)-Xn(6:hm5+1,5:wm5)+ Xnp1(5:hm5,6:wm5+1)-Xn(5:hm5,6:wm5+1) +Xnp1(5:hm5,5:wm5)-Xn(5:hm5,5:wm5))/4;
dsx2(5:hm5,5:wm5) = ( Xnp1(6:hm5+1,6:wm5+1)-Xnp1(5:hm5,6:wm5+1) + Xnp1(6:hm5+1,5:wm5)-Xnp1(5:hm5,5:wm5)+ Xn(6:hm5+1,6:wm5+1)-Xn(5:hm5,6:wm5+1) +Xn(6:hm5+1,5:wm5)-Xn(5:hm5,5:wm5))/4;
dsx1(5:hm5,5:wm5) = ( Xnp1(6:hm5+1,6:wm5+1)-Xnp1(6:hm5+1,5:wm5) + Xnp1(5:hm5,6:wm5+1)-Xnp1(5:hm5,5:wm5)+ Xn(6:hm5+1,6:wm5+1)-Xn(6:hm5+1,5:wm5) +Xn(5:hm5,6:wm5+1)-Xn(5:hm5,5:wm5))/4;

for i=1:imax
   delta=(dsx1.*v1+dsx2.*v2+dst)./(alpha2+dsx1.^2+dsx2.^2);
   v1=v1-dsx1.*delta;
   v2=v2-dsx2.*delta;
end;
u=z; u(5:hm5,5:wm5)=v1(5:hm5,5:wm5);
v=z; v(5:hm5,5:wm5)=v2(5:hm5,5:wm5);

xskip=round(h/64);
[hs,ws]=size(u(1:xskip:h,1:xskip:w));
us=zeros(hs,ws); vs=us;

N=xskip^2;
for i=1:hs-1
  for j=1:ws-1
     hk=i*xskip-xskip+1;
     hl=i*xskip;
     wk=j*xskip-xskip+1;
     wl=j*xskip;
     us(i,j)=sum(sum(u(hk:hl,wk:wl)))/N;
     vs(i,j)=sum(sum(v(hk:hl,wk:wl)))/N;
   end;
end;

figure(1);
quiver(us,vs);
colormap('default');
axis ij;
axis tight;
axis equal;

end