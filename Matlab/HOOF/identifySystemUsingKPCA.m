function sysParams = identifySystemUsingKPCA(Yoriginal,kernelMatrix, order)

% sysParams = identifySystemUsingKPCA(Yoriginal, kernelMatrix, order)
%
% Takes as input the kernel computed on the output trajectory of a sequence
% and computes the relevant system parameters as mentioned in A.B.Chan CVPR
% 07
% (c) Rizwan Chaudhry - JHU Vision Lab

K = double(kernelMatrix);
N = size(kernelMatrix,1);

e = ones(1,N)';

KTilde = (eye(N)-e*e'/N)*K*(eye(N)-e*e'/N);

% remove any numerical inconsistencies by making KTilde symmetric

KTilde = (KTilde + KTilde')/2;

[V,D] = eig(KTilde);

d = real(diag(D))';

% Format data such that the eigen values and corresponding eigen vectors
% are listed in descending order

V = V(:,end:-1:1);
d = d(end:-1:1);

alpha = V./sqrt(repmat(d,size(V,1),1));

X = alpha'*KTilde;

X = X(1:order,:);
alpha = alpha(:,1:order);

A = X(:,2:end)*pinv(X(:,1:end-1));
V = zeros(order,N);
V(:,2:end) = X(:,2:end)-A*X(:,1:end-1);
Q = zeros(size(V,1),size(V,1));
for i=1:N-1
    Q = Q + V(:,i)*V(:,i)';
end
Q = 1/(N-1)*Q;

% Y - minimum norm reconstruction
% R

sysParams.A = A;
sysParams.alpha = alpha;
sysParams.X = X;
sysParams.K = K;
sysParams.KTilde = KTilde;
sysParams.Yoriginal = Yoriginal;
sysParams.Q = Q;
sysParams.order = order;