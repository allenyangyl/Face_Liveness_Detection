function angles = findSubspaceAnglesBetweenKPCASystems(sysParams1, sysParams2)

% angles = findSubspaceAnglesBetweenKPCASystems(sysParams1,sysParams2)
%
% Finds subspace angles between two systems identified using KPCA
%
% (c) Rizwan Chaudhry - JHU Vision Lab

Y_12 = [sysParams1.Yoriginal, sysParams2.Yoriginal];

% Evaluate the kernel on these vectors;
kernel_12 = zeros(size(Y_12,2));

for i=1:length(Y_12)
   for j =1:i-1
      % Choose appropriate kernel here
       kernel_12(i,j) = computeDistanceModularHistogram(Y_12(:,i),Y_12(:,j));  % MDPA
   end
end

kernel_12 = kernel_12 + kernel_12';

kernel_12 = exp(-kernel_12.^2);

N1 = size(sysParams1.Yoriginal,2);
N2 = size(sysParams2.Yoriginal,2);

e1 = ones(1,N1)';
e2 = ones(1,N2)';

alphaPrime1 = sysParams1.alpha-1/N1*(repmat(sum(sysParams1.alpha,1),size(sysParams1.alpha,1),1));
alphaPrime2 = sysParams2.alpha-1/N2*(repmat(sum(sysParams2.alpha,1),size(sysParams2.alpha,1),1));

F_11 = alphaPrime1'*kernel_12(1:N1,1:N1)*alphaPrime1;
F_12 = alphaPrime1'*kernel_12(1:N1,N1+1:end)*alphaPrime2;
F_21 = alphaPrime2'*kernel_12(N1+1:end,1:N1)*alphaPrime1;
F_22 = alphaPrime2'*kernel_12(N1+1:end,N1+1:end)*alphaPrime2;

M_11 = dlyap(sysParams1.A',sysParams1.A,F_11);
M_12 = dlyap(sysParams1.A',sysParams2.A,F_12);
M_21 = dlyap(sysParams2.A',sysParams1.A,F_21);
M_22 = dlyap(sysParams2.A',sysParams2.A,F_22);

lambda = eig(pinv(M_11)*M_12*pinv(M_22)*M_21);
lambda = real(lambda);
lambda = max(zeros(size(lambda)),lambda);
lambda = min(ones(size(lambda)),lambda);
lambda = sort(lambda,'descend');

lambda = sqrt(lambda);

order = min(sysParams1.order,sysParams2.order);

angles = real(acos(lambda(1:order)));