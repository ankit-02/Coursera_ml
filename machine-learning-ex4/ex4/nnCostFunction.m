function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

Xnew = [ones(m, 1) X];
middle_layer_z = Theta1 * Xnew' ;
middle_layer = sigmoid(middle_layer_z) ;
middle_layer_trans = middle_layer' ;
middle_layer_new = [ones(m, 1) middle_layer_trans];
output_layer = sigmoid(Theta2 * middle_layer_new') ;
% max(y)
yNew=[zeros(max(y),size(y,1))];
% yNew=[zeros(10,size(y,1))];
for i = 1:size(y,1)
    yNew(y(i,1),i) = 1;
endfor;
J = - ( (log(output_layer).*yNew) + (log(1-output_layer).* (1-yNew)) ) ;
% Jj= - ( (log(output_layer).*yNew));
% J= - ( (log(output_layer).*yNew));
% J-= ((log(1-output_layer).* (1-yNew)) ) ;

J_reg = sum(sum(Theta1(:,2:end).^2));
% J_reg = sum(sum(Theta1(:,2:size(Theta1,2)).^2));
J_reg += sum(sum(Theta2(:,2:end).^2));
% J_reg += sum(sum(Theta2(:,2:size(Theta2,2)).^2));

J = sum(sum(J))/m + J_reg*lambda/(2*m);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%

% delta_j_three = output_layer - yNew;
% fprintf("ankit\n\n");
% % size(delta_j_three); #10,5000
% % size(Theta2) #10,26
% % size(delta_j_three) #10,5000 
% delta_j_second =  Theta2' * delta_j_three  ;
% size(delta_j_second) #26,5000
% size(middle_layer_z) # 25,5000
% delta_j_second = delta_j_second.*   sigmoidGradient(middle_layer_z);

for i = 1:size(y,1)
    xt= [1,X(i,:)];
    z2= Theta1 * xt' ;
    a2= [1;sigmoid(z2)];
    z3= Theta2 * a2 ;
    a3 = sigmoid(z3);
    delta_j_three = a3 - yNew(:,i);
    delta_j_second = Theta2'*delta_j_three .* [1;sigmoidGradient(z2)];
    delta_j_second = delta_j_second(2:end);
    Theta1_grad+=delta_j_second*xt;
    Theta2_grad+=delta_j_three*a2';
endfor;

Theta1_grad += lambda*[zeros(size(Theta1,1),1),Theta1(:,2:end)];
Theta2_grad += lambda*[zeros(size(Theta2,1),1),Theta2(:,2:end)];
Theta1_grad/=m;
Theta2_grad/=m;



%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
