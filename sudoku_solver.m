function [X0, dual] = sudoku_solver(soudoku, X_init)

given = zeros(0,3);
for i=1:9
    for j=1:9
        if soudoku(i,j)>0
            given = [given; i, j, soudoku(i,j)];
        end
    end
end

rho = 1000;
epsilon = 1e-28;
T = 200000;
I = eye(9);
X0 = zeros(9,9,9);
X1 = zeros(9,9,9);
lam1 = zeros(9,9,9);
X2 = zeros(9,9,9);
lam2 = zeros(9,9,9);
X3 = zeros(9,9,9);
lam3 = zeros(9,9,9);
X4 = zeros(9,9,9);
lam4 = zeros(9,9,9);
X5 = zeros(9,9,9);
lam5 = zeros(9,9,9);
X6 = zeros(9,9,9);
lam6 = zeros(9,9,9);
dual1 = zeros(T,1);
dual2 = zeros(T,1);
dual3 = zeros(T,1);
dual4 = zeros(T,1);
dual5 = zeros(T,1);
dual6 = zeros(T,1);
dual = zeros(T,1);
primal = zeros(T,1);

%% init
s = size(given);
el = s(1);
% X0 = X_init;
for i=1:el
    X0(:, given(i,1), given(i,2)) = I(:,given(i,3));
end
% X0 = randn(9,9,9);
X1 = X0;
X2 = X0;
X3 = X0;
X4 = X0;
X5 = X0;
X6 = X0;

for it=1:T
    %% update X0
    X0p = X0;
%     %% Convex
%     X0 = (1/6).*(X1+X2+X3+X4+X5+X6)+(1/(6*rho)).*(lam1+lam2+lam3+lam4+lam5+lam6);
    
    %% non-convex 2
    X0 = (1/6).*(X1+X2+X3+X4+X5+X6)+(1/(6*rho)).*(lam1+lam2+lam3+lam4+lam5+lam6);
    A = 6*rho.*X0;
    for newtone_it=1:5
%         (pi*cos(pi*x0)+6*rho*x0-a)
        X0 = X0 - (pi.*cos(pi.*X0)+ 6*rho.*X0-A)./(-pi^2.*sin(pi.*X0)+6*rho);
    %     x0 = x0-.05*(pi*cos(pi*x0)+6*rho*x0-a);
    end
    
%     %% non-convex-1
%     uu = 2;
%     vv = 1;
%     X0 = (rho/(6*rho-uu)).*(X1+X2+X3+X4+X5+X6)+(1/(6*rho-uu)).*(lam1+lam2+lam3+lam4+lam5+lam6-vv);
%     


    primal(it) = sum(sum(sum((X0p-X0).^2)));
    
    
    
    %% updating X1 (0<=X1<=1)
    X1 = X0-(1/rho).*lam1;
    X1 = max(0,X1);
    X1 = min(1,X1);
    %% updating X2 (row constraint)
    X2 = X0-(1/rho).*lam2;
    for i=1:9
        for k=1:9
            s = sum(sum(sum(X2(k, i, :))));
            X2(k, i, :) = (1-s)/9+X2(k, i, :);
        end
    end
    %% updating X3 (col constraint)
    X3 = X0-(1/rho).*lam3;
    for j=1:9
        for k=1:9
            s = sum(sum(sum(X3(k, :, j))));
            X3(k, :, j) = (1-s)/9+X3(k, :, j);
        end
    end
    %% updating X4 (Box constraints)
    X4 = X4-(1/rho).*lam4;
    for i=1:3:7
        for j=1:3:7
            for k=1:9
                s = sum(sum(sum(X4(k, i:i+2, j:j+2))));
                X4(k, i:i+2, j:j+2) = (1-s)/9+X4(k, i:i+2, j:j+2);
                %sum(sum(sum(B.*(X.^2)))) <= 1;
            end
        end
    end
    %% updating X5 (cell constraint)
    X5 = X0-(1/rho).*lam5;
    for i=1:9
        for j=1:9
            s = sum(sum(sum(X5(:, i, j))));
            X5(:, i, j) = (1-s)./9+X5(:, i, j);
        end
    end
    %% updating X6 (given constraints)
    X6 = X0-(1/rho).*lam6;
    s = size(given);
    el = s(1);
    for i=1:el
        X6(:, given(i,1), given(i,2)) = I(:,given(i,3));
    end
    
    

    %% Update dual variables
    lam1 = lam1 + rho.*(X1-X0);
    lam2 = lam2 + rho.*(X2-X0);
    lam3 = lam3 + rho.*(X3-X0);
    lam4 = lam4 + rho.*(X4-X0);
    lam5 = lam5 + rho.*(X5-X0);
    lam6 = lam6 + rho.*(X6-X0);
    
    %% Computing the duals
    dual1(it) = sum(sum(sum((X1-X0).^2)));
    dual2(it) = sum(sum(sum((X2-X0).^2)));
    dual3(it) = sum(sum(sum((X3-X0).^2)));
    dual4(it) = sum(sum(sum((X4-X0).^2)));
    dual5(it) = sum(sum(sum((X5-X0).^2)));
    dual6(it)= sum(sum(sum((X6-X0).^2)));
    dual(it) = dual1(it)+dual2(it)+dual3(it)+dual4(it)+dual5(it)+dual6(it);
    if dual(it)<epsilon && primal(it)<epsilon
        break
    end
        
end

figure(1)
title('dual')
semilogy(dual)
hold on
figure(2)
title('primal')
semilogy(primal)
hold on
end