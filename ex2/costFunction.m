function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

    
theta1 = theta';
[row,col] = size(theta1);

for j = 1:col
    J2 = 0;
    for i = 1:m
        XX = X(i:i,:);
        JJ = XX * theta;
        J1 = 1 / (1 + exp(-JJ));
        J2 = J2 + (J1 - y(i))*X(i,j);
    end
    J2 = J2 / m;
    grad(j,1) = J2;
end


for i = 1:m
    XX = X(i:i,:);
    JJ = XX * theta;
    J1 = 1 / (1 + exp(-JJ));
    J2 = ( -y(i)*log(J1) )  - ( (1-y(i))*log(1-J1) );
    J = J + J2;
end

J = J / m;

        





% =============================================================

end
