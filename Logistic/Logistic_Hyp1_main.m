close all
clc
%% Initialization
training_data=heartDD; %Imported in a cell array format
target=cell2mat(training_data(:,14)); %target
parameters=cell2mat((training_data(:,1:3))); %parameters from 4 to the 7

%% Normalization
target=target/max(target); %Normalizing output
for k=1:1:size(parameters,2)
parameters(:,k)=parameters(:,k)/max(parameters(:,k)); %normalizing inputs
end
%% Adding ones column to the parameters
parameters=[ones(length(parameters),1) parameters];
%% Attributes for the Algorithm
m=length(target); %Quantity of training data
thetas=randn(size(parameters,2),1); %Weights of the parameters 
alpha=0.01; %Learning rate
iterations=3000; %Number of iterations until reaching a best fit line
mse=[]; %MSEs
i=[]; %iterations

%% Algorithm output
 for ii=1:1:iterations
     
   %% Hypothesis
   hyp=hypo_log(parameters,thetas);
   %% Cost function
   cost=cost_fn_log(hyp,target,m);
   mse=[mse;cost];
   i=[i;ii];
   %% Gradient descent Algorithm
   thetas_new= gradient_desc(parameters,m,hyp,target,thetas,alpha);
   %% Updating Thetas
   thetas_prev=thetas;
   
   thetas=thetas_new;
 end
 hyp=hypo_log(parameters,thetas); %final hypothesis
 
%% Iterations vs Mean Square Error for Linear regression
 figure (1)
 plot(i,mse)

 %% Stochastic filtering (Normal equation)
 H=inv(parameters'*parameters)*(parameters'*target);
 hypoth=hypo_log( parameters,H );
 mse2=cost_fn_log( hypoth,target,m );
 
 %% Testing data
 testing_data=heartDDtest;
 testing_parameters=cell2mat((testing_data(:,1:3)));
 testing_target=cell2mat(testing_data(:,14));
 m1=length(testing_target);
 %% Testing data normalization
 testing_target_norm=testing_target/max(testing_target); %Normalizing output
for r=1:1:size(testing_parameters,2)
testing_parameters(:,r)=testing_parameters(:,r)/max(testing_parameters(:,r)); %normalizing inputs
end
%% Adding ones column to the testing parameters
testing_parameters=[ones(length(testing_parameters),1) testing_parameters];
 
 hyp_test=hypo_log(testing_parameters,thetas);
 cost_test=cost_fn_log( hyp_test,testing_target_norm,m1 );