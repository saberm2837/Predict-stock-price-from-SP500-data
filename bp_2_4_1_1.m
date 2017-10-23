function bp_2_4_1_1( )
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
    P_train = P_train';
    T_train = T_train';
    P_test = P_test';
    T_test = T_test';
    P_valid = P_valid';
    T_valid = T_valid';
    % initial weigh and bias
    w1_0 = [-0.27 0.45; -0.48 0.35; 0.55 0.65; 0.34 0.23];
    b1_0 = [-0.48; -0.13; -0.56; 0.59];
    w2_0 = [0.09 -0.17 0.45 0.78];
    b2_0 = [0.48];
    w3_0 = [0.89];
    b3_0 = [0.55];
    
    for(j=1:12000)
        % training
        for(i=1:20)
            p = P_train(:,i); 
            a0 = p;  
            a1 = logsig(w1_0*p + b1_0);  
            a2 = logsig(w2_0*a1 + b2_0);
            a3 = purelin(w3_0*a2 + b3_0);
            e = T_train(:,i) - a3;

            s3 = -e*1;
            g2 = [(1-a2)*a2];
            s2 = g2*w3_0'*s3;
            g1 = [(1-a1(1))*(a1(1)) 0 0 0; 0 (1-a1(2))*(a1(2)) 0 0; 0 0 (1-a1(3))*(a1(3)) 0; 0 0 0 (1-a1(4))*(a1(4))];
            s1 = g1*w2_0'*s2;

            lr = 0.1;
            w3_1=w3_0 - lr*s3*a2';
            b3_1=b3_0 - lr*s3;  
        
            w2_1=w2_0 - lr*s2*a1';
            b2_1=b2_0 - lr*s2;  

            w1_1=w1_0 - lr*s1*a0';
            b1_1=b1_0 - lr*s1;
            
            w1_0 = w1_1;
            b1_0 = b1_1;
            w2_0 = w2_1;
            b2_0 = b2_1;
            w3_0 = w3_1;
            b3_0 = b3_1;
        end
        % testing
        err = [];
        for(i=1:10)
            p = P_test(:,i); 
            a0 = p;  
            a1 = logsig(w1_0*p + b1_0);  
            a2 = logsig(w2_0*a1 + b2_0);
            a3 = purelin(w3_0*a2 + b3_0);
            e = T_test(:,i) - a3;
            err = [err e^2];
        end
        MSE = sum(err);
    end
    err = [];
    for(i=1:10)
        p = P_valid(:,i); 
            a0 = p;  
            a1 = logsig(w1_0*p + b1_0);  
            a2 = logsig(w2_0*a1 + b2_0);
            a3 = purelin(w3_0*a2 + b3_0);
            e = T_valid(:,i) - a3;
            err = [err e^2];
    end
    MSE = sum(err)
end