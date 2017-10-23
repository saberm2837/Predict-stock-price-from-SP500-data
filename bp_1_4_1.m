function bp_1_4_1( )
% creating ,training, testing & validation set for y = 1/x
    y = inline('1/x');
    x1 = 0.1; 
    x2 = 1;
    total_data = 40;
    step = (x2-x1)/(total_data-1);
    P_train = [];
    T_train = [];
    P_test = [];
    T_test = [];
    P_valid = [];
    T_valid = [];
    
    for(i=1:total_data)
        if(mod(i,2)==1)     % training data set
            P_train = [P_train; x1+step*(i-1)];
            T_train = [T_train; y(x1+step*(i-1))];
        end

        if(mod(i,4)==2)     % testing data set
            P_test = [P_test; x1+step*(i-1)];
            T_test = [T_test; y(x1+step*(i-1))];
        end
        
        if(mod(i,4)==0)     % validation data set
            P_valid = [P_valid; x1+step*(i-1)];
            T_valid = [T_valid; y(x1+step*(i-1))];
        end
    end
    % initial weigh and bias
    w1_0 = [-0.27; -0.41; -0.61; -0.34];
    b1_0 = [-0.48; -0.13; -0.34; -0.23];
    w2_0 = [0.09  -0.17 0.12 -0.11];
    b2_0 = [0.48];
    
    for(j=1:500)
        % training
        lr = 0.01;
        for(i=1:20)
            p = P_train(i); 
            a0 = p;  
            a1 = logsig(w1_0*p + b1_0);  
            a2 = purelin(w2_0*a1 + b2_0);
            e = T_train(i) - a2;

            s2 = -e*1;
            g1 = [(1-a1(1))*(a1(1)) 0 0 0; 0 (1-a1(2))*(a1(2)) 0 0; 0 0 (1-a1(3))*(a1(3)) 0; 0 0 0 (1-a1(4))*(a1(4))];
            s1 = g1*w2_0'*s2;

            w2_1=w2_0 - lr*s2*a1';
            b2_1=b2_0 - lr*s2;  

            w1_1=w1_0 - lr*s1*a0';
            b1_1=b1_0 - lr*s1;
            
            w1_0 = w1_1;
            b1_0 = b1_1;
            w2_0 = w2_1;
            b2_0 = b2_1;
        end
        % testing
        err = [];
        for(i=1:10)
            p = P_test(i); 
            a0 = p;  
            a1 = logsig(w1_0*p + b1_0);  
            a2 = purelin(w2_0*a1 + b2_0);
            e = T_test(i) - a2;
            err = [err e^2];
        end
        MSE = sum(err);
    end
    err = [];
    for(i=1:10)
        p = P_valid(i); 
        a0 = p;  
        a1 = logsig(w1_0*p + b1_0);  
        a2 = purelin(w2_0*a1 + b2_0);
        e = T_valid(i) - a2;
        err = [err e^2];
    end
    MSE = sum(err)
end