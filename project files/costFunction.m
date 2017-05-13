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
%alpha=0.01;


J=-(1/m)*sum(y'*log(sigmoid(X*theta))+(1-y)'*log(1-sigmoid(X*theta)));

%J=-(1/m)*sum(y'*log(sigmoid(X*theta))+(1-y)'*log(1-sigmoid(X*theta)));
%for i=1:1500

grad=1/m*X'*(sigmoid(X*theta)-y);
%end
%grad = (1 / m) * sum( X .* repmat((sigmoid(X*theta) - y), 1, size(X,2)) );
%theta0=0;
%theta1=0;
%theta2=0;
%for i=1:20000
%theta0=theta0-1/m*X'*(sigmoid(X*theta)-y);
%theta1=theta1-1/m*X'*(sigmoid(X*theta)-y);
%theta2=theta2-1/m*X'*(sigmoid(X*theta)-y);
%grad=[theta0; theta1; theta2];
%end




end
