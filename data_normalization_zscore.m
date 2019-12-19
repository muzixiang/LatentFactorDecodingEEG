%% 对每个被试各个电极的数据按照z-score进行normalize并且重构封装格式

subNum = 32;
trialNum = 40;
channelNum = 40;
fs = 128;
trialTime = 63; 
trialL = fs*trialTime;

for subNo=1:subNum
    %norm_data = zeros(trialNum,signalL);
    if subNo<10
        filePath = strcat('D:\DEAP DATA\s0',num2str(subNo),'.mat');
    else
        filePath = strcat('D:\DEAP DATA\s',num2str(subNo),'.mat');
    end
    datFile = load(filePath);
    subData = datFile.data;
    reshape_subData = zeros(channelNum, trialNum*trialL);
    for channelNo = 1:channelNum
        for trialNo = 1:trialNum
            ch_tr_signal = subData(trialNo,channelNo,:);
            reshape_subData(channelNo,(trialNo-1)*trialL+1:trialNo*trialL)=ch_tr_signal;
        end
    end
    
    zscore_data = zscore(reshape_subData');
    
    %将该被试的data保存起来
    fileName = strcat('D:\Processed DEAP DATA\normalize_zscore\sub',num2str(subNo));
    save(fileName,'zscore_data','-v7.3');
    disp(strcat('ends!subject ',num2str(subNo)));
end