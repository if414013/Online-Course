function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



p = X*theta;
temp2 = (1/(2*m))*(p - y).^2;
theta2 = theta(2:end);
regular = (lambda/(2*m))* sum(theta2.^2);
J = sum(temp2)+regular;
% =========================================================================

%tampa regularisasi
grad = 1 / m * X' * (p - y);
%untuk dua dan selebihnya tambahkan regularisasi
grad(2:end) = grad(2:end) + lambda / m * theta(2:end);
grad = grad(:);

end
