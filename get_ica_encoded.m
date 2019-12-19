
subNum=32;
channelNum=32;
zscore=1;

latdim = 16;

for subNo=1:subNum
        zscore_eegs_file = load(strcat('D:\Processed DEAP DATA\normalize_zscore\sub',num2str(subNo),'.mat')); 
        zscore_eegs = zscore_eegs_file.zscore_data(:,1:channelNum)';
        disp(strcat('subNo: ',num2str(subNo),' latentdim: ', num2str(latdim)));
         %fast ICA A:Mixing matrix W: Seperating matrix
        [ICs, A, W] = fastica(zscore_eegs,'numOfIC',latdim,'verbose','off');
        decoded_eegs = A*ICs;
        encoded_eegs = ICs;
        %save data
        fileName = strcat('D:\VAE Experiment\DEAP\encoded_eegs_ica\encoded_eegs_ica_sub',num2str(subNo),'_latedtdim',num2str(latdim));
        save(fileName,'encoded_eegs','-v7.3');
end
