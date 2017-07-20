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
grad

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

p = X*theta;
p = sigmoid(p);
temp = (-y.*log(p)) - ((1-y).*(log(1-p))) + (lambda/(2*m))*sum(theta(2:length(theta)).*theta(2:length(theta)));
J = (1/m)*sum(temp);
    
for j=1:size(X,2)
    if j==1
        tot = (1/m)*(p - y).*  X(:,j);
        grad(j) = sum(tot);
    else
        tot = (1/m)*(p - y).*  X(:,j);
        grad(j) = sum(tot) + (lambda/m)*theta(j);
    end
end

% J = ( (1 / m) * sum(-y'*log(sigmoid(X*theta)) - (1-y)'*log( 1 - sigmoid(X*theta))) ) + (lambda/(2*m))*sum(theta(2:length(theta)).*theta(2:length(theta))) ;
% 
% grad = (1 / m) * sum( X .* repmat((sigmoid(X*theta) - y), 1, size(X,2)) );
% 
% grad(:,2:length(grad)) = grad(:,2:length(grad)) + (lambda/m)*theta(2:length(theta))';


% =============================================================

end
