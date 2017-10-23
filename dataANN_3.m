function [P T] = dataANN_3( )
% creating ,training, testing & validation set for y = 1/x
    y=inline('1/x');
    P = [];
    T = [];
    x1 = 0.1;
    x2 = 1;
    step = (x2-x1)/39;
    for(x = x1 : step : x2)
        P = [P x];
        T = [T y(x)];
    end
end