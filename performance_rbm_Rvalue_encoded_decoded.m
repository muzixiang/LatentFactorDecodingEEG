subNum=32;
channelNum=32;


corr_chs = zeros(subNum,channelNum);
for subNo=1:subNum
    filename1 = strcat(strcat('D:\VAE Experiment\decoded_eegs_rbm\decoded_eegs_rbm_sub',num2str(subNo),'_latentdim',num2str(latent_dims(subNo)),'_interdim',num2str(inter_dims(subNo)),'.mat'));
    decoded_eegs_rbm_file = load(filename1);
    decoded_eegs_rbm = decoded_eegs_rbm_file.decoded_eegs;
    zscore_eegs_file = load(strcat('D:\Processed DEAP DATA\normalize_zscore\sub',num2str(subNo),'.mat'));
    zscore_eegs = zscore_eegs_file.zscore_data;
    for chno=1:channelNum
        R=corrcoef(decoded_eegs_rbm(:,chno), zscore_eegs(:,chno));
        corr_chs(subNo,chno)=R(1,2);
    end
    corr_chs(subNo,:)
end
mean(corr_chs)