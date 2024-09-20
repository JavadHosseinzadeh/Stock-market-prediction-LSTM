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
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Normalization%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creating Input & Output
data=[Close movingaverage MACDLine MACDSignalLine FastPercentK FastPercentD SlowPercentK SlowPercentD upperBolling lowerBolling];
data=data(35:end,:);
% scaling

%%
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
%network
numTimeStepsTrain = floor(0.98*(size(data,1)));
dataTrain = data(1:numTimeStepsTrain+1,:);
dataTest = data(numTimeStepsTrain+1:end,:);
numFeatures = 10;
numHiddenUnits1 = 200;
numHiddenUnits2 = 100;
numHiddenUnits3 = 75;
numHiddenUnits4 = 50;
numHiddenUnits5 = 25;
numResponses=10;
numClasses = 9;
layers = [ ...
    sequenceInputLayer(numFeatures)

    lstmLayer(numHiddenUnits1,'OutputMode','sequence')
    dropoutLayer(0.2)

    fullyConnectedLayer(numFeatures)
    regressionLayer
    ];
%%
numofepoch=500;
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
[net,YPred] = predictAndUpdateState(net,dataTrain(2:end,:).');
[net,dd] = predictAndUpdateState(net,data.','ExecutionEnvironment','cpu');
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
%% plot of network:
figure,
plot(1:length(YPred),YPred)
hold on
plot(1:length(Ytrain),Ytrain)
figure,
plot(1:length(dd),dd)
hold on
plot(1:length(target),target)
figure,
plot(4149:length(dd),dd(4149:end))
hold on
plot(4149:length(target),target(4149:end))
