function [Z,U,evals] = PCA(X,K)
  
% X is N*D input data, K is desired number of projection dimensions (assumed
% K<D).  Return values are the projected data Z, which should be N*K, U,
% the D*K projection matrix (the eigenvectors), and evals, which are the
% eigenvalues associated with the dimensions
  
[N D] = size(X);

if K > D,
  error('PCA: you are trying to *increase* the dimension!');
end;

% first, we have to center the data

%TODO
mean_data = zeros(1,D);
for i = 1:N
	mean_data = mean_data + X(i,:);
end
mean_data = mean_data/N;

for i=1:N
	X(i,:) = X(i,:) - mean_data;
end

% next, compute the covariance matrix C of the data

%TODO
% C = cov(X);
C = (X'*X)/N;

% compute the top K eigenvalues and eigenvectors of C... 
% hint: you may use 'eigs' built-in function of MATLAB



%TODO
U = zeros(D,K);
[V,E] = eig(C);
Eigen = zeros(D,1);
for i = 1:D
	Eigen(i,:) = E(i,i);
end
evals = zeros(1,K);
for i = 1:K
	[eigenvalue, index] = max(Eigen);
	Eigen(index,:) = 0;
	evals(:,i) = eigenvalue;
	U(:,i) = V(:,index);

end


% project the data

Z = X*U;
