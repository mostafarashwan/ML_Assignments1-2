function [ hyp ] = hypo_log( x,thetas )

hyp=1./(1+exp(-x*thetas));


end

