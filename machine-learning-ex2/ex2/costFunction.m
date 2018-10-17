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

% calculate sigmoid
sigmoidVal = sigmoid(X * theta);

% for y = 1
p1y1 = log(sigmoidVal);
firstPart = -y .* p1y1;

% for y = 0
p2y0 = log(1 - sigmoidVal);
secondPart = (1 - y) .* p2y0; % second part of the equation --> y = 0

J = sum(firstPart - secondPart)/m;

% for gradient
part1 = sigmoidVal - y;
grad = sum(part1.*X)/m;


% =============================================================

end
