load ('mnist.mat') ;
Euclidian_Norm = zeros(10000, 10) ;
Y_Predicted = zeros(10000,1);
Accuracy = []
Train_count = []

for i = 1:40
	correct_count = 0;
    Mean = [];
	for j = 1:10
			Mean = [Mean; mean(dataX{j}(1:50*i, :))];
	end

	for k = 1:10000
		for l = 1:10
			Euclidian_Norm(k,l) = norm(X_test(k,:) - Mean(l, :));
		end

	end

	for m = 1:10000
		[M,I] = min(Euclidian_Norm(m, :));
		Y_Predicted(m,1) = I-1 ;
	end

	correct_count = sum(Y_Predicted == Y_test) ;

	Accuracy = [Accuracy, correct_count/100]
    Train_count = [Train_count, 50*i] ;
end

plot(Train_count, Accuracy)