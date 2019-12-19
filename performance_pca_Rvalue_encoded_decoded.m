subNum=32;
channelNum=32;
testNum=10;
latdimNum=16;

corr_chs = zeros(subNum,latdimNum,channelNum);


    for subNo=1:subNum
        zscore_eegs_file = load(strcat('D:\Processed DEAP DATA\normalize_zscore\sub',num2str(subNo),'.mat'));
        zscore_eegs = zscore_eegs_file.zscore_data(:,1:32)';
        for latdim = 1:latdimNum
            disp(strcat(' subNo: ', num2str(subNo), ' latdim: ', num2str(latdim)));         
            [residuals, reconstructed] = pcares(zscore_eegs, latdim);
            decoded_eegs_pca = reconstructed;
            for chno1=1:channelNum
                max_r = 0;
                for chno2=1:channelNum
                     R=corrcoef(decoded_eegs_pca(chno1,:), zscore_eegs(chno2,:));
                     r=R(1,2);
                     if r>max_r
                         max_r=r;
                     end    
                end
                corr_chs(subNo,latdim,chno1)=max_r;
            end
            corr_chs(subNo,latdim, :)
        end
    end

%mean(corr_chs)