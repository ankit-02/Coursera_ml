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

% size(X)
% size(y)
% size(theta)
predict_y= X * theta;
diff_y= predict_y - y;

J= sum(diff_y.^2(:)) + lambda *(sum(theta .^2(:))- theta(1,1)^2 );
J/=(2 * m);


% grad(1,:)=sum(diff_y .* X(:,1));
% grad(2:end,:)=sum(diff_y .* X)
grad = X' * diff_y + lambda * theta;
grad(1,1)-=lambda*theta(1,1);
grad/=m;






% =========================================================================

grad = grad(:);

end
