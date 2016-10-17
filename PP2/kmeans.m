function [mu,z,score] = kmeans(X,K,init)
% input X is N*D data, K is number of clusters desired.  init is either
% 'random' which just chooses K random points to initialize with, or
% 'furthest', which chooses the first point randomly and then uses the
% "furthest point" heuristic.  the output is mu, which is K*D, the
% coordinates of the means, and z, which is N*1, and only integers 1...K
% specifying the cluster associated with each data point.  score is the
% score of the clustering
  
[N D] = size(X);

if K >= N,
  error('kmeans: you are trying to make too many clusters!');
end;

if nargin<3 || strcmp(init, 'random'),
  % initialize by choosing K (distinct!) random points: we do this by
  % randomly permuting the examples and then selecting the first K
  perm = randperm(N);
  perm = perm(1:K);
  
  % the centers are given by points
  mu = X(perm, :);
  
  % we leave the assignments unspecified -- they'll be computed in the
  % first iteration of K means
  z = zeros(N,1);

elseif strcmp(init, 'furthest'),
  % initialize the first center by choosing a point at random; then
  % iteratively choose furthest points as the remaining centers

  %TODO
  perm = randperm(N);
  perm = perm(1:1);
  mu(1,:) = X(perm,:);
  for i = 1:size(X,1)
    dist(i,:) = norm(X(i,:)-mu(1,:));
  end
  for i = 2:K
    [M,index] = max(dist);
    mu(i,:) = X(index,:);
    dist(index,:) = 0;
  end
  % again, don't bother initializing z
  z = zeros(N,1);
else
  error('unknown initialization: use "furthest" or "random"');
end;

% begin the iterations.  we'll run for a maximum of 20, even though we
% know that things will *eventually* converge.
for iter=1:20,
  % in the first step, we do assignments: each point is assigned to the
  % closest center.  we'll judge convergence based on these assignments,
  % so we want to keep track of the previous assignment
  
  oldz = z;
  
  for n=1:N,
    % assign point n to the closest center
    for i = 1:K
      cent_dist(i,:) = norm(X(n,:) - mu(i,:));
    end
    [m,z(n,:)] = min(cent_dist);
    %TODO
  end;
  
  % check to see if we've converged
  if all(oldz==z),
    break;  % break out of loop
  end;
  
  
  % re-estimate the means
  for i = 1:K
    summation = zeros(1,size(mu',1));
    set_size = 0;
    for j = 1:N
      if z(j,:) == i
        summation = summation + X(j,:);
        set_size = set_size + 1;
      end
    end
    mu(i,:) = summation/set_size;
  end
  %TODO
end;

% final: compute the score
score = 0;
for n=1:N,
  % compute the distance between X(n,:) and it's associated mean
  score = score + norm(X(n,:) - mu(z(n),:))^2;
end;
