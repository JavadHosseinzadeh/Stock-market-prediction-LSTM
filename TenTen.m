%% in the name of god 
% Seminar project for stock prediction
clc
clear all
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Data Extraction%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% loading data & visualize:
stocks = hist_stock_data('23012003','07262021','Gold');
%%
Time=stocks.Date;
Open=stocks.Open;
High=stocks.High;
Low=stocks.Low;
Close=stocks.Close;
Volume=stocks.Volume;
DataBase_Table=table(Open,High,Low,Close,Volume);
DataBase=timetable(Time,Open,High,Low,Close,Volume);
%% indicators:
% Moving average
% character vector with value of 'simple', 'square-root', 'linear', 'square', 'exponential', 'triangular', 'modified', or 'custom' | string with value of "simple", "square-root", "linear", "square", "exponential", "triangular", "modified", or "custom"
windowSize=13;
movingaverage = movavg(Close,'linear',windowSize);
% MACD
[MACDLine,MACDSignalLine] = macd(Close);
% bollinger
[middleBolling,upperBolling,lowerBolling]= bollinger(Close,'WindowSize',21);
% stochastic ocsilation
oscillator = stochosc(DataBase,'NumPeriodsD',7,'NumPeriodsK',10,'Type','exponential');
FastPercentK=table2array(oscillator(:,1));
FastPercentD=table2array(oscillator(:,2));
SlowPercentK=table2array(oscillator(:,3));
SlowPercentD=table2array(oscillator(:,4));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Network%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Creating Input & Output
Data=[stocks.Close...
       movingaverage MACDLine ...
       MACDSignalLine upperBolling lowerBolling...
       FastPercentK FastPercentD SlowPercentK SlowPercentD
];
Data=Data(100:end,:);
target=stocks.Close(100:end,:);
%% normalize data 
%                            (z-mu)
%                            ------
%                              sig
numsam = size(Data,1);
mu = mean(Data,1);
sig = std(Data,1);
MU = repmat(mu,numsam,1);
SIG = repmat(sig,numsam,1);
data = (Data-MU)./SIG;
%% network
numTimeStepsTrain = floor(0.98*(size(data,1)));
dataTrain = data(1:numTimeStepsTrain,:);
dataTest = data(numTimeStepsTrain+1:end,:);
Ytrain=target(1:numTimeStepsTrain,:);
Ytest=target(numTimeStepsTrain+1:end,:);
numFeatures = 10;
numHiddenUnits1 = 150;
numHiddenUnits2 = 100;
numHiddenUnits3 = 50;
numHiddenUnits4 = 20;
numHiddenUnits5 = 10;
numResponses=10;
numClasses = 9;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits1,'OutputMode','sequence')
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits2)
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits3)
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits4)
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits5)
    dropoutLayer(0.2)
    fullyConnectedLayer(numResponses)
    regressionLayer
    ];
%%
numofepoch=300;
options = trainingOptions('adam', ...
    'MaxEpochs',numofepoch, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',1,'Shuffle', 'never', ...
    'Plots','training-progress');
net = trainNetwork(dataTrain(1:end-1,:).',dataTrain(2:end,:).',layers,options);
%%
%net = trainNetwork(dataTrain.',Ytrain.',layers,options);
%[net,YPred] = predictAndUpdateState(net,dataTrain(2:end,:).');

[net,dd] = predictAndUpdateState(net,dataTrain.','ExecutionEnvironment','cpu');

% for i=1:size(dataTest,1)
%     [net,dd(:,size(dataTrain,1)+i)] = predictAndUpdateState(net,(dd(:,size(dataTrain,1)+i-1)).','ExecutionEnvironment','cpu');
% end


%%
Predict=dd(1,:)'.*sig(1,1)+mu(1,1);
figure,
plot(2:size(Data,1)+1,Data(:,1))
hold on
plot(1:size(Predict,1),Predict)
legend('Data','Predict')
perform(net,Data(1:size(dd,2),:),dd')
perf = mse(net,Data(1:size(dd,2),:),dd')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Ploting%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot dataset of Gold
figure('Name','data')
subplot(211)
plot(stocks.Date,stocks.High,'LineWidth' , 2)
hold on 
plot(stocks.Date,stocks.Low,'LineWidth' , 2)
hold on
plot(stocks.Date,stocks.Close,'LineWidth' , 2)
xlabel('Date (day)')
ylabel('price ($)')
legend('High Value','Low Value','Close Value')
title('Symbol : Gold')
subplot(212)
plot(stocks.Date,stocks.Volume,'LineWidth' , 2)
xlabel('Date (day)')
ylabel(' Volume')
legend('Trade Volume')
%% plot Moving average of Gold symbol
figure('Name','Moving average')
plot(stocks.Date,movingaverage,'LineWidth' , 2)
xlabel('Date (day)')
ylabel('Moving average')
legend('MA window=13')
title('Symbol : Gold')
%% plot MACD of Gold
figure('Name','MACD')
plot(stocks.Date,MACDLine,'LineWidth' , 2)
hold on
plot(stocks.Date,MACDSignalLine,'LineWidth' , 2)
grid on
xlabel('Date (day)')
ylabel('MACD Indicator')
legend('MACD Line','MACD Signal Line')
title('Symbol : Gold')
%% plot Stochastic of Gold
figure('Name','Stochastic oscillator')
%subplot(511)
plot(stocks.Date,oscillator(:,1),'LineWidth' , 2)
hold on
plot(stocks.Date,oscillator(:,2),'LineWidth' , 2)
hold on
plot(stocks.Date,oscillator(:,3),'LineWidth' , 2)
xlabel('Date (day)')
ylabel('price ($)')
legend('MACD Line')
title('Symbol : Gold')
%% plot bollinger of Gold
figure('Name','Bollinger Band')
%[middleBolling,upperBolling,lowerBolling]
plot(stocks.Date,middleBolling,'LineWidth' , 2)
hold on
plot(stocks.Date,upperBolling,'LineWidth' , 2)
hold on
plot(stocks.Date,lowerBolling,'LineWidth' , 2)
xlabel('Date (day)')
ylabel('Bollinger band indicator')
legend('middle Bolling','upper Bolling','lower Bolling')
title('Symbol : Gold')
%% plot Stochastic of Gold
figure('Name','Stochastic Oscillator')
%[middleBolling,upperBolling,lowerBolling]
subplot(211)
plot(Time,FastPercentD,'LineWidth' , 2)
hold on
plot(Time,FastPercentK,'LineWidth' , 2)
xlabel('Date (day)')
ylabel('Fast Stochastic Oscillator')
legend('%Fast Percent D','%Fast Percent K')
title('Symbol : Gold')
subplot(212)
plot(Time,SlowPercentD,'LineWidth' , 2)
hold on
plot(Time,SlowPercentK,'LineWidth' , 2)
xlabel('Date (day)')
ylabel('Fast Stochastic Oscillator')
legend('%Slow Percent D','%Slow Percent K')

































