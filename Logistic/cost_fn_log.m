function [ cost ] = cost_fn( hyp,price,m )

cost= (1/(2*m))*sum((hyp-price).^2);


end
