function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
%���ϵ�һ��1����ΪX��m * n�ģ�ÿ��������������һ�У����Լ��ϵľ��� m*1��һ�У�����ÿһ�е�ǰ������˸�1
X = [ones(m, 1) X];
for i = 1 : m
    XX = X(i:i,:);
    z2 = Theta1 * XX';
    z2 = sigmoid(z2);
    %row = size(z2,1)
    %���ϵ�һ�� 1����Ϊz2�� n*1�ģ�����i��������z2���ǵ���һ�У��Ǹ��������������Ҫ����һ��1��������һ��1
    z22 = [ones(1,1);z2];
    
    z3 = Theta2 * z22;
    a3 = sigmoid(z3);
    [maxInd, index] = max(a3,[],1);
    if index == 0
        index = 10;
    end
    p(i) = index;
end;
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================


end
