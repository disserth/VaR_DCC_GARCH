

clc;
clear all;
tic;


Price=xlsread('VaR Automobile.csv','VaR Automobile','B:K'); 
Value=xlsread('VaR Automobile.csv','VaR Automobile','L:L');


alpha95=1.65;
alpha99=2.33;


[T,n]=size(Price);
Return=log(Price(2:T,:)./Price(1:(T-1),:)); % calculate stock return
[T,n]=size(Return);
W=ones(1,n)*(1/n);
% W=Price(1,:)./Value(1);

PL=Value(2:T,1)-Value(1:(T-1),1);    % dimension (T-1)*1

t=70;   % estimation horizon / moving window size
Residuals=zeros(t,n);
VaR=zeros(T-t,2);

for j=1:(T-t)
    
    %%%%%%  DCC-GARCH   %%%%%%%%
   
    dccP=1;
    dccQ=1;
    
    % get the residuals from AR(1)
    Temp=Return(j:(j+(t-1)),:);
    
    
    for i=1:n
     
        [parAR,LL,Residuals(:,i)]=armaxfilter(Temp(:,i),0,1,0);
  
    end
    
    
%      try;
    [parameters, loglikelihood, Ht_dcc]=dcc(Residuals,[],dccP,dccQ, []);
    D_dcc(:,:,j)=Ht_dcc(:,:,t);   % conditional covariance matrix modeled by DCC-GARCH
    VaR(j,1)=sqrt(W*D_dcc(:,:,j)*W')*alpha95*Value(j);    % 95 percent Value-at-Risk
    VaR(j,2)=sqrt(W*D_dcc(:,:,j)*W')*alpha99*Value(j);    % 99 percent Value-at-Risk
%     catch;
%         %if previous loop fails vor one VaR use previous value
%     VaR(j,1)=VaR(j-1,1);    % 95 percent Value-at-Risk
%     VaR(j,2)=VaR(j-2,2);    % 99 percent Value-at-Risk
%     end;
end 

PL_Test=PL(t:T-1); 

%backtesting results for 5% risk
[rst1,ES5,ViolationRatio5]=BacktestChristoffensen(PL_Test, VaR(:,1), 0.05);
%backtesting results for 1% risk
[rst5,ES1,ViolationRatio1]=BacktestChristoffensen(PL_Test, VaR(:,2), 0.01);
%  output: statistics and pvalue of Backtest;   
% Input:PL, VaR and significant level
toc;