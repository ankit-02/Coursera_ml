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

Cs=[0.01,0.03,0.1,0.3,1,2,10,30];
sigmas=[0.01,0.03,0.1,0.3,1,2,10,30];
Cs_index=-1;
sigma_index=-1;
lowest_error=-1;
for css = 1:8
    for sigmass = 1:8
        model= svmTrain(X, y, Cs(1,css), @(x1, x2) gaussianKernel(x1, x2, sigmas(1,sigmass))); 
        predic = svmPredict(model,Xval);
        here_error=mean(double(predic~=yval));
        if(lowest_error == -1)
            sigma_index=sigmass;
            Cs_index=css;
            lowest_error = here_error;
        else if (lowest_error > here_error)
            sigma_index=sigmass;
            Cs_index=css;
            lowest_error = here_error;
        endif;

    end
end

C = Cs(1,Cs_index);
sigma=sigmas(1,sigma_index);




% =========================================================================

end
