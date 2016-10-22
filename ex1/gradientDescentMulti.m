function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
	theta1 = theta;
	for j = 1 : size(theta,1)
		J = 0.0;
		for i = 1 : m
			JJ = 0.0;
			for k = 1 : size(theta,1)
				JJ = JJ + theta(k) * X(i,k);
			end
			JJ = (JJ - y(i)) * X(i,j);
			J = J + JJ;
		end
		theta1(j) = theta(j) - alpha * J / m;
	end
	theta = theta1;
    J_history(iter) = computeCostMulti(X, y, theta);
end





    % ============================================================

    % Save the cost J in every iteration    
    

  

end
