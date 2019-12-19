
subNum=32;
channelNum=32;
zscore=1;

latdim = 16;

for subNo=1:subNum
        zscore_eegs_file = load(strcat('D:\Processed DEAP DATA\normalize_zscore\sub',num2str(subNo),'.mat')); 
        zscore_eegs = zscore_eegs_file.zscore_data(:,1:channelNum)';
        disp(strcat('subNo: ',num2str(subNo),' latentdim: ', num2str(latdim)));
        
        X = zscore_eegs.';
        [coeff, score, latent] = princomp(X);
        encoded_eegs = score(:,1:latdim);
%         [pcaData, coeff] = fastPCA(X,latdim);
%         encoded_eegs = pcaData;
        %save data
        fileName = strcat('D:\VAE Experiment\DEAP\encoded_eegs_pca\encoded_eegs_pca_sub',num2str(subNo),'_latedtdim',num2str(latdim));
        save(fileName,'encoded_eegs','-v7.3');
end
