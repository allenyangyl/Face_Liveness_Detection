function kernel = traceKernelKPCASystems(sysParams1, sysParams2, lambda)

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

F_12 = alphaPrime1'*kernel_12(1:N1,N1+1:end)*alphaPrime2;

M_12 = dlyap(lambda*sysParams1.A',sysParams2.A,F_12);
M_12 = real(M_12);

B1 = real(chol(sysParams1.Q,'lower'));
B2 = real(chol(sysParams2.Q,'lower'));

kernel = sysParams1.X(:,1)'*M_12*sysParams2.X(:,1) + lambda/(1-lambda)*trace(B1'*M_12*B2);