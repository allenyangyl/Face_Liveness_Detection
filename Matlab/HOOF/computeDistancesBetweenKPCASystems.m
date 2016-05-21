function distances = computeDistancesBetweenKPCASystems(sysParams,metric,kernel)

% function distances =
% computeDistancesBetweenKPCASystems(sysParams,metric,kernel)
%
% Computes the required kernel distance
%
% (c) Rizwan Chaudhry - JHU Vision Lab
%

kernelName{1} = 'Geodesic';
kernelName{2} = 'MDPA-ordinal';
kernelName{3} = 'ChiSquare';
kernelName{4} = 'HIST';

metricName{1} = 'Martin';
metricName{2} = 'BC-trace';
metricName{3} = 'BC-init-ind';

lambda = 0.9; % Parameter for Binet Cauchy kernels

% Compute all pairwise sequence kernels
%disp('Computing all pairwise sequence kernels');
for ind_i = 1:length(sysParams)
    for ind_j = 1:ind_i
        Y_12 = double([sysParams{ind_i}.Yoriginal, sysParams{ind_j}.Yoriginal]);
        % Evaluate the kernel on these vectors;
        kernel_12 = zeros(size(Y_12,2),'double');

        % Evaluate the kernel on these vectors;

        if strcmp(kernelName{kernel}, 'Geodesic')
            Y_12 = sqrt(Y_12);
            kernel_12 = Y_12'*Y_12;
        elseif strcmp(kernelName{kernel}, 'MDPA-ordinal')
            for i=1:size(Y_12,2)
               for j =1:i-1
                   kernel_12(i,j) = computeDistanceOrdinalHistogram(Y_12(:,i),Y_12(:,j));  % MDPA
               end
            end
            kernel_12 = kernel_12 + kernel_12';
            kernel_12 = exp(-(kernel_12.^2));
        elseif strcmp(kernelName{kernel}, 'ChiSquare')
            kernel_12 = exp(-(chiSquareDist(Y_12).^2));
        elseif strcmp(kernelName{kernel}, 'HIST')
            for i=1:size(Y_12,2)
                for j = 1:i
                    kernel_12(i,j) = histogramKernel(Y_12(:,i),Y_12(:,j));
                end
            end
            kdiag = diag(kernel_12);
            kernel_12 = kernel_12 + kernel_12' - diag(kdiag);
        end
        
        allPairWiseKernels{ind_i,ind_j} = kernel_12;
        
    end
end

distances = zeros(length(sysParams),length(sysParams),length(metric));

for m = 1:length(metric)
   if strcmp(metricName{metric(m)},'Martin') == 1
       
       %disp('Martin distance');
       
       for ind_i = 1:length(sysParams)
           for ind_j = 1:ind_i
               sysParams1 = sysParams{ind_i};
               sysParams2 = sysParams{ind_j};
               kernel_12 = allPairWiseKernels{ind_i,ind_j};
       
                N1 = size(sysParams1.Yoriginal,2);
                N2 = size(sysParams2.Yoriginal,2);

                order = min(sysParams1.order,sysParams2.order);
                
%                 alphaPrime1 = sysParams1.alpha;
%                 alphaPrime2 = sysParams2.alpha;
                alphaPrime1 = sysParams1.alpha-1/N1*(repmat(sum(sysParams1.alpha,1),size(sysParams1.alpha,1),1));
                alphaPrime2 = sysParams2.alpha-1/N2*(repmat(sum(sysParams2.alpha,1),size(sysParams2.alpha,1),1));

                F_11 = alphaPrime1'*kernel_12(1:N1,1:N1)*alphaPrime1;
                F_12 = alphaPrime1'*kernel_12(1:N1,N1+1:end)*alphaPrime2;
                F_21 = alphaPrime2'*kernel_12(N1+1:end,1:N1)*alphaPrime1;
                F_22 = alphaPrime2'*kernel_12(N1+1:end,N1+1:end)*alphaPrime2;
%                 
%                 M_11 = dlyap(sysParams1.A',sysParams1.A,F_11);
%                 M_12 = dlyap(sysParams1.A',sysParams2.A,F_12);
%                 M_21 = dlyap(sysParams2.A',sysParams1.A,F_21);
%                 M_22 = dlyap(sysParams2.A',sysParams2.A,F_22);
% 
%                 evals = eig(pinv(M_11)*M_12*pinv(M_22)*M_21);
%
%                 angles = real(acos(sqrt(evals(1:order))));
                

                Ft = [F_11,F_12;F_21,F_22];                
                At = [sysParams1.A,zeros(order);zeros(order),sysParams2.A];                
                
                M = dlyap(At',Ft);
                evals = eig([zeros(order) pinv(M(1:order,1:order))*M(1:order,order+1:2*order);...
                    pinv(M(order+1:2*order,order+1:2*order))*M(order+1:2*order,1:order) zeros(order)]);
                
                evals = real(evals);
                evals = max(-ones(size(evals)),evals);
                evals = min(ones(size(evals)),evals);
                evals = sort(evals,'descend');
                
                angles = real(acos(evals(1:order)));
                
                distances(ind_i,ind_j,m) = sqrt(-sum(log((cos(angles)).^2)));
                
                distances(ind_j,ind_i,m) = distances(ind_i,ind_j,m);
           end
       end
       
       
       
       
   elseif strcmp(metricName{metric(m)},'BC-trace') == 1
       
       %disp('Binet Cauchy Kernel');       
    
        for ind_i = 1:length(sysParams)
           for ind_j = 1:ind_i
               sysParams1 = sysParams{ind_i};
               sysParams2 = sysParams{ind_j};
               kernel_12 = allPairWiseKernels{ind_i,ind_j};

               N1 = size(sysParams1.Yoriginal,2);
               N2 = size(sysParams2.Yoriginal,2);

               alphaPrime1 = sysParams1.alpha-1/N1*(repmat(sum(sysParams1.alpha,1),size(sysParams1.alpha,1),1));
               alphaPrime2 = sysParams2.alpha-1/N2*(repmat(sum(sysParams2.alpha,1),size(sysParams2.alpha,1),1));

               F_12 = alphaPrime1'*kernel_12(1:N1,N1+1:end)*alphaPrime2;

               M_12 = dlyap(lambda*sysParams1.A',sysParams2.A,F_12);
               M_12 = real(M_12);

               % Check for numerical errors due to very small eigenvalues
               % of Q
%                [V1,D1] = eig(sysParams1.Q);
%                if any(diag(D1) <= 0)
%                    diag1 = diag(D1);
%                    diag1(diag1 <= 0) = min(abs(diag1));
%                    D1 = diag(diag1);
%                    sysParams1.Q = V1*D1*V1';
%                end
%                [V2,D2] = eig(sysParams2.Q);
%                if any(diag(D2) <= 0)                   
%                    diag2 = diag(D2);
%                    diag2(diag2 <= 0) = min(abs(diag2));
%                    D2 = diag(diag2);
%                    sysParams2.Q = V2*D2*V2';
%                end

               % Regularize Qs with 1
               sysParams1.Q = sysParams1.Q + eps*eye(size(sysParams1.Q));
               sysParams2.Q = sysParams2.Q + eps*eye(size(sysParams2.Q));               
               
               B1 = real(chol(sysParams1.Q))';
               B2 = real(chol(sysParams2.Q))';

               kernel = sysParams1.X(:,1)'*M_12*sysParams2.X(:,1) + lambda/(1-lambda)*trace(B1'*M_12*B2);

               distances(ind_i,ind_j,m) = kernel;
               distances(ind_j,ind_i,m) = kernel;
           end
        end

   elseif strcmp(metricName{metric(m)},'BC-init-ind') == 1
       
       %disp('Initial state independent Binet Cauchy Kernel');       

        for ind_i = 1:length(sysParams)
           for ind_j = 1:ind_i
               sysParams1 = sysParams{ind_i};
               sysParams2 = sysParams{ind_j};
               kernel_12 = allPairWiseKernels{ind_i,ind_j};

               N1 = size(sysParams1.Yoriginal,2);
               
               N2 = size(sysParams2.Yoriginal,2);

               alphaPrime1 = sysParams1.alpha-1/N1*(repmat(sum(sysParams1.alpha,1),size(sysParams1.alpha,1),1));
               alphaPrime2 = sysParams2.alpha-1/N2*(repmat(sum(sysParams2.alpha,1),size(sysParams2.alpha,1),1));

               F_12 = alphaPrime1'*kernel_12(1:N1,N1+1:end)*alphaPrime2;

               M_12 = dlyap(lambda*sysParams1.A',sysParams2.A,F_12);
               M_12 = real(M_12);
               
               kernel = max(svd(M_12));

               distances(ind_i,ind_j,m) = kernel;
               distances(ind_j,ind_i,m) = kernel;
           end
        end
        
   end
   
end