function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

d = [.01;.03;.1;.3;1;3;10;30];

eVec = [];
pVec = [];

for q1 = 1:8
    for q2 = 1:8
        
         mi= svmTrain(X, y, d(q1), @(x1, x2) gaussianKernel(x1, x2, d(q2)));
        
         p_i = svmPredict(mi, Xval);
         e_i = mean(double(p_i ~= yval));
      
         pVec = [pVec; [d(q1),d(q2)]];
         eVec = [eVec;e_i];
    end
end

[mY,mI] = min(eVec)
C = pVec(mI,1);
sigma = pVec(mI,2);


% =========================================================================

end
