function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
    if j >= 2
        grad(j,1) = J2 + (lambda / m ) * theta(j);
    else
        grad(j,1) = J2;
    end
end


for i = 1:m
    XX = X(i:i,:);
    JJ = XX * theta;
    J1 = 1 / (1 + exp(-JJ));
    J2 = ( -y(i)*log(J1) )  - ( (1-y(i))*log(1-J1) );
    J = J + J2;
end

J = J / m;
J = J + (lambda / 2 / m ) * ( sum(theta.^2) - theta(1)^2);



% =============================================================

end
