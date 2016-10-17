% Compute predictions for the samples in test_X based on the training data in train_X (samples) and train_y (labels)
% k = number of neighbours
% distance_metric = Euclidean or ...
function predictions = knn(train_X, train_y, test_X, k, distance_metric)
	m = size(train_y); % Number of samples
	n = size(train_X, 2); % Number of features
	predictions = zeros(size(test_X, 1), 1);

	for i=1:size(test_X)
		test_sample_matrix = ((test_X(i, :))' * ones(1, m))'; % Create a matrix of m rows, where each row contains the current test sample -> makes computing the distances easier
		size((test_X(i, :))')
		size(ones(1, m))
		size(test_sample_matrix)
		distances = sum((train_X .- test_sample_matrix) .^ 2, 2);
		
		% Find the min k distances
		mins = ones(k, 1); % Stores the indices of training samples which are closest to the test sample
		for j=1:k
			[min pos] = min(distances);
			mins(j) = pos;
			distances(pos) = max(distances);
		endfor
		
		min_labels = train_y(mins);
		
		% Decide class of sample based on majority class of k selected neighbours
		% unique + find
		unique_labels = unique(min_labels);
		max = 0;
		for j=1:k
			crt_vote = sum(min_labels == unique_labels(j));
			if crt_vote>max
				max = crt_vote;
				predicted_label = unique_labels(j);
			endif
		endfor
		
		predictions(i) = predicted_label;
	endfor
end