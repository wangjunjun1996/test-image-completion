clear;
%定义7*4*3的张量，7代表7个流量检测器，4代表4天，3代表三个时间窗（每15min）
% obs(:,:,1) = [155,74,493,426;108,44,350,359;
%     175,78,567,581;181,111,517,552;
%     137,53,489,485;90,44,306,290;
%     139,55,398,390];
% obs(:,:,2) = [172,69,590,386;104,39,310,304;
%     158,74,505,546;176,90,525,552;
%     150,64,438,459;73,32,281,299;
%     127,51,358,382];
% obs(:,:,3) = [225,92,443,436;94,44,355,356;
%     139,77,575,604;175,98,574,553;
%     126,67,593,484;58,49,348,301;
%     144,71,444,396];
% [n1,n2,n3] = size(obs);

obs=imread('e:\test\tomato.jpg'); 
%imshow(obs);
[n1,n2,n3] = size(obs);
obs1 = double(obs);

%随机生成一个大小为7*4*3的稀疏张量S，元素为0或1，张量S与obs点乘即可得到新的张量X
%有一部分数据缺失，缺失程度可根据如下程序自行调整
S = round(rand(n1,n2,n3)+0.3);
X = S.*obs1;
X1=uint8(X);
imwrite(X1,'missing tomato.JPG'),;% 保存帧
%imshow('missing apple.JPG');
pos_obs = find(X~=0); % index set of observed entries
pos_unobs = find(X==0); % index set of missing entries

%设置参数alpha,rho,max
alpha = ones(1,3)./3;
%rho = 10^(-2);
rho1=10^(-2);
maxiter = 1000;
t=ones(1,maxiter);

%变量初始化
X_hat = X;
Y1 = zeros(n1,n2,n3); Y2 = Y1; Y3=Y1; % additive tensor

%进行迭代更新
convergence = zeros(maxiter,1);
for iter = 1:maxiter
    iter
    [u1,s1,v1] = svds(ten2mat(X_hat,[n1,n2,n3],1)+ten2mat(Y1,[n1,n2,n3],1)/rho1, n1);
    B1 = mat2ten(u1*diag(max(diag(s1)-alpha(1)/rho1,0))*v1',[n1,n2,n3],1); % update tensor B1
    [u2,s2,v2] = svds(ten2mat(X_hat,[n1,n2,n3],2)+ten2mat(Y2,[n1,n2,n3],2)/rho1, n2);
    B2 = mat2ten(u2*diag(max(diag(s2)-alpha(2)/rho1,0))*v2',[n1,n2,n3],2); % update tensor B2
    [u3,s3,v3] = svds(ten2mat(X_hat,[n1,n2,n3],3)+ten2mat(Y3,[n1,n2,n3],3)/rho1, n3);
    B3 = mat2ten(u3*diag(max(diag(s3)-alpha(3)/rho1,0))*v3',[n1,n2,n3],3); % update tensor B3
    X_hat = (1-S).*(B1+B2+B3-(Y1+Y2+Y3)/rho1)/3+S.*X; % update the estimated tensor
    Y1 = Y1-rho1*(B1-X_hat);
    Y2 = Y2-rho1*(B2-X_hat);
    Y3 = Y3-rho1*(B3-X_hat);
    t1=t+0.0001;
    rho1=rho1*t1(iter);
    convergence(iter,1) = sum(X_hat(pos_unobs).^2);
end

X_hat1=uint8(X_hat);
imwrite(X_hat1,'recovering tomato.JPG'),;
%imshow('recovering apple.JPG')
err=sum(abs(obs1(pos_unobs)-X_hat(pos_unobs)))./sum(X_hat(pos_unobs));
figure(1); 
subplot(2,2,1);imshow(obs),title('原始图像'); %显示原始图像
subplot(2,2,2);imshow('missing tomato.JPG'),title('丢失图像'); 
subplot(2,2,3);imshow('recovering tomato.JPG'),title('恢复图像');
figure(2);
plot(convergence);
%RMSE = sqrt(sum((obs(pos_unobs)-X_hat(pos_unobs)).^2)./length(pos_unobs));
%MRE = sum(abs(obs(pos_unobs)-X_hat(pos_unobs)))./sum(X_hat(pos_unobs));
fprintf('\n--------------HaLRTC-------------------\n')
%fprintf('RMSE = %g veh/15min, MRE = %g', RMSE, MRE);
fprintf('err = %g ', err);
fprintf('\n---------------------------------------\n')