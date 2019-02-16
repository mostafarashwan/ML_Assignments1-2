close all
clc
%% Initialization
training_data=housepricesdatatrainingdata; %Imported in a cell array format
price=cell2mat(training_data(:,3)); %Price
parameters=cell2mat((training_data(:,4:8))); %parameters from 4 to the 7

%% Normalization
price=(price-mean(price))/std(price); %Normalizing output
for k=1:1:size(parameters,2)
parameters(:,k)=(parameters(:,k)-mean(parameters(:,k)))/(std(parameters(:,k))); %normalizing inputs
end
%% Adding ones column to the parameters
parameters=[ones(length(parameters),1) parameters];
%% Attributes for the Algorithm
m=length(price); %Quantity of training data
thetas=randn(size(parameters,2),1); %Weights of the parameters 
alpha=0.001; %Learning rate
iterations=50000; %Number of iterations until reaching a best fit line
mse=[]; %MSEs
i=[]; %iterations

%% Algorithm output
 for ii=1:1:iterations
     
   %% Hypothesis
   hyp=hypo(parameters,thetas);
   %% Cost function
   cost=cost_fn(hyp,price,m);
   mse=[mse;cost];
   i=[i;ii];
   %% Gradient descent Algorithm
   thetas_new= gradient_desc(parameters,m,hyp,price,thetas,alpha);
   %% Updating Thetas
   thetas_prev=thetas;
   
   thetas=thetas_new;
 end
 hyp=hypo(parameters,thetas); %final hypothesis
 
%% Iterations vs Mean Square Error for Linear regression
 figure (1)
 plot(i,mse)

 %% Stochastic filtering (Normal equation)
 H=inv(parameters'*parameters)*(parameters'*price);
 hypoth=hypo( parameters,H );
 mse2=cost_fn( hypoth,price,m );
 
 %% Testing data
 testing_data=housedatacomplete;
 testing_parameters=cell2mat((testing_data(:,4:8)));
 testing_price=cell2mat(testing_data(:,3));
 m1=length(testing_price);
 %% Testing data normalization
 testing_price_norm=(testing_price-mean(testing_price))/std(testing_price); %Normalizing output
for r=1:1:size(testing_parameters,2)
testing_parameters(:,r)=(testing_parameters(:,r)-mean(testing_parameters(:,r)))/(std(testing_parameters(:,r))); %normalizing inputs
end
%% Adding ones column to the testing parameters
testing_parameters=[ones(length(testing_parameters),1) testing_parameters];
 
 hyp_test=hypo(testing_parameters,thetas);
 cost_test=cost_fn( hyp_test,testing_price_norm,m1 );