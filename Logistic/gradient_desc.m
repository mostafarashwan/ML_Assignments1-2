function [ thetas_new ] = gradient_desc( x,m,hyp,price,thetas,alpha )

thetas_new=[];
for i=1:1:size(x,2)
diff=(1/m)*sum((hyp-price).*x(:,i));
thetas_new1=thetas(i,1)-(alpha*diff);
thetas_new=[thetas_new;thetas_new1];


end

