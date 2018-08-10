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
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
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

X = [ones(m, 1) X];

zl2 = Theta1 * X';
al2 = sigmoid(zl2)';
al2 = [ones(m, 1) al2];

zl3 =  al2 * Theta2';
al3 = sigmoid(zl3);

[a1,a2] = max(al3,[],2);

y_MClass = zeros(m,num_labels);

for t = 1 :m
   y_MClass(t,y(t)) = 1;
end    




for k = 1:num_labels
  s1 = al3(:,k);
  J = J + (-1/m)*(   y_MClass(:,k)' * log(s1) +    (1-y_MClass(:,k)')*log(1-s1)) ;
end  
  
rSum1 = 0;
rSum2 = 0;
for k2 = 1:hidden_layer_size
  rSum1 = rSum1 + sum(Theta1(k2,2:end).^2);
end    

for k3 = 1:num_labels
  rSum2 = rSum2 + sum(Theta2(k3,2:end).^2);
end    



reg = (1/(2*m))*lambda * (rSum1 + rSum2) ;
  
J = J + reg;
% -------------------------------------------------------------

delta_l3 = al3;

D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

for i = 1:m

 al1i = X(i,:);
    
 zl2i = Theta1 * al1i';
 al2i_nb = sigmoid(zl2i)';
 al2i = [ones(1,1) al2i_nb]';

 zl3i =  Theta2 * al2i;
 al3i = sigmoid(zl3i);  
    
 d3 = al3i - y_MClass(i,:)'  ;
  
 d2 = (Theta2' * d3) ;
 d2 = d2(2:end) .* sigmoidGradient(zl2i);
      
 D2 = D2 + d3 * al2i'; % 10 x 26
 D1 = D1 + d2 * al1i; % 25 x 401
 
end    

%Theta1_grad = 1/m*D1 ;
%Theta2_grad = 1/m*D2 ;

Theta1_grad = 1/m*D1 + lambda/m*[zeros(size(Theta1,1),1) Theta1(:, 2:end)];
Theta2_grad = 1/m*D2 + lambda/m*[zeros(size(Theta2,1),1) Theta2(:, 2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
