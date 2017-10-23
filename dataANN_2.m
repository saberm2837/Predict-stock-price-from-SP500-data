function [P_train,T_train,P_test,T_test,P_valid,T_valid] = dataANN_2( )
% creating ,training, testing & validation set for z = x^2+y^2
    z = inline('x^2+y^2');
    x1 = -1; 
    x2 = 1;
    y1 = -1; 
    y2 = 1;
    total_data = 40;
    step1 = (x2-x1)/(total_data-1);
    step2 = (y2-y1)/(total_data-1);
    P_train = [];
    T_train = [];
    P_test = [];
    T_test = [];
    P_valid = [];
    T_valid = [];
    
    for(i=1:total_data)
        if(mod(i,2)==1)     % training data set
            P_train = [P_train; [x1+step1*(i-1) y1+step2*(i-1)]];
            T_train = [T_train; z(x1+step1*(i-1),y1+step2*(i-1))];
        end

        if(mod(i,4)==2)     % testing data set
            P_test = [P_test; [x1+step1*(i-1) y1+step2*(i-1)]];
            T_test = [T_test; z(x1+step1*(i-1),y1+step2*(i-1))];
        end
        
        if(mod(i,4)==0)     %validation data set
            P_valid = [P_valid; [x1+step1*(i-1) y1+step2*(i-1)]];
            T_valid = [T_valid; z(x1+step1*(i-1),y1+step2*(i-1))];
        end
    end
end