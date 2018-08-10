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



for i = 1:m
    htxi = sigmoid(theta' * X(i,:)');
    J = J + (1/m)*(-y(i)*log(htxi) - (1-y(i))*log(1-htxi) ) ;
end

for j = 2:size(theta)
    J = J + (1/(2*m))*lambda * theta(j)^2;
end

htx = sigmoid(theta' * X');

grad(1) = ((1/m) * sum(htx' - y));

for j = 2:size(theta)
   grad(j) = ((1/m) * (X(:,j)' * (htx' - y))) + (lambda/m)*theta(j);
end






% =============================================================

end
