function [P T] = dataANN_4( )
% creating ,training, testing & validation set for z = x^2+y^2
    z = inline('x^2+y^2');
    x1 = -1; 
    x2 = 1;
    y1 = -1; 
    y2 = 1;
    total_data = 40;
    step1 = (x2-x1)/(total_data-1);
    step2 = (y2-y1)/(total_data-1);
    P = [];
    T = [];
    for(i=1:total_data)
        P = [P; [x1+step1*(i-1) y1+step2*(i-1)]];
        T = [T; z(x1+step1*(i-1),y1+step2*(i-1))];
    end
end