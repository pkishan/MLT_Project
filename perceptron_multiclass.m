% You are only suppose to complete the "TODO" parts and the code should run
% if these parts are correctly implemented
clear all;close all;
load('mnist_big.mat');
[N D] = size(X_train);
Y_train = Y_train + 1;
Y_test = Y_test + 1;
W = randn(D,max(Y_train));
b = randn(1,max(Y_train));
W_avg = W;
b_avg = b;
k = 0; % number of mistakes (i.e., number of updates)

% Just do a single pass over the training data (our stopping condition will
% simply be this)
C= X_train';
D = X_test';
for n=1:N
    % TODO: predict the label for the n-th training example
    x_n = C(:, n);
    A = W'*x_n + b';
    [maxL, ind] = max(A(:));
    y_pred = ind;
    if y_pred ~= Y_train(n)
        k = k + 1;
        % TODO: Update W, b
        W(:,Y_train(n)) = W(:, Y_train(n)) + x_n;
        b(Y_train(n)) = b(Y_train(n)) + 1;
        W(:,y_pred) = W(:, y_pred) - x_n;
        b(y_pred) = b(y_pred) - 1;
        % TODO: Update W_avg, b_avg using Ruppert Polyak Averaging
        % Important: You don't need to store all the previous W's and b's
        % to compute W_avg, b_avg
        W_avg = W_avg - (W_avg - W)/k;
        b_avg = b_avg - (b_avg - b)/k;
        
        % TODO: Predict test labels using W, b
        for i = 1:size(Y_test, 1)
            x_i =  D(:,i); 
            B = W'*x_i + b';
            [maxA ,y_test_pred(i,1)] = max(B(:));   % ?? TODO
        end
        acc(k) = mean(Y_test==y_test_pred);   % test accuracy     
        
        % TODO: Now predict test labels using W_avg, b_avg
        for i = 1:size(Y_test, 1)
            x_i = D(:,i);
            B = W_avg'*x_i + b_avg';
            [maxA ,y_test_pred(i,1)] = max(B(:)) ;   % ?? TODO
        end
        acc_avg(k) = mean(Y_test==y_test_pred); % test accuracy with R-P averaging
       
        fprintf('Update number %d, accuracy = %f, accuracy (with R-P averaging) = %f\n',k,acc(k),acc_avg(k));
    end
end
plot(1:k,acc(1:k),'r');
hold on;
plot(1:k,acc_avg(1:k),'g');
drawnow;
