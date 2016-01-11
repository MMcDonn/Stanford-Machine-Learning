function [theta, J_history, iterations, tol] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, size(X,2)); %initializes cost function 
                                         %progression vector
iterations = 0;

%Compute gradient descent minimization on the theta function. Then plug in
%the updated theta function into the cost function
for i = 1:num_iters
    theta = theta - (alpha/m)* X' * (X*theta - y);   
    J_history(i) = computeCost(X, y, theta);
    iterations = iterations + 1;
    
    if i > 1 %gives condition so that norm indicies are valid
        %Measures convergence. If norm of consecutive costs functions is 
        %changing by less than 0.00001
        tol = norm(J_history(i,:) - J_history(i-1,:));
        if tol < 10^(-5) 
            break;
        end
    end
end

end
