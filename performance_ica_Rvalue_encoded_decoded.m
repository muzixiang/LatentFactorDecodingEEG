
latent_dims = 1:1:16;
%inter_dims = [16, 64, 128, 256];
subNum=32;
channelNum=32;
zscore=1;

corr_chs = zeros(subNum,length(latent_dims),channelNum);

for subNo=1:subNum
    for j=1:length(latent_dims)
        disp(strcat('subNo: ',num2str(subNo),' latentdim: ', num2str(latent_dims(j))));
        zscore_eegs_file = load(strcat('D:\Processed DEAP DATA\normalize_zscore\sub',num2str(subNo),'.mat'));
        zscore_eegs = zscore_eegs_file.zscore_data(:,1:32)';
         %fast ICA A:Mixing matrix W: Seperating matrix
        [ICs, A, W] = fastica(zscore_eegs,'numOfIC',latent_dims(j),'verbose','off');
        decoded_eegs = A*ICs;
        %decoded_eegs_ica = A*(W*zscore_eegs);
        
        for chno1=1:channelNum
            max_r = 0;
            for chno2=1:channelNum
                 R=corrcoef(decoded_eegs(chno1,:), zscore_eegs(chno2,:));
                 r=R(1,2);
                 if r>max_r
                     max_r=r;
                 end    
            end
            corr_chs(subNo,j,chno1)=max_r;
        end
%         for chno=1:channelNum
%             R=corrcoef(decoded_eegs(chno,:), zscore_eegs(chno,:));
%             corr_chs(subNo,j,chno)=R(1,2);
%         end
    end
end

%各个被试表现最佳的结果及其对应的网络结构
best_sub_corrs = zeros(1,subNum);
best_latent_dims = zeros(1,subNum);
for subNo=1:subNum
    sub_corr_chs = squeeze(corr_chs(subNo,:,:));
    mean_sub_corr_chs = mean(sub_corr_chs,2);
    [best_sub_corr,best_latent_dim] = max(mean_sub_corr_chs);
    best_sub_corrs(1,subNo)=best_sub_corr;
    best_latent_dims(1,subNo)=best_latent_dim;
end
best_sub_corrs
mean_best_sub_corr = mean(best_sub_corrs);
best_latent_dims
mean_best_sub_corr

%平均被试表现最佳的结果及其对应的网络结构
latentdim_mean_corrs = zeros(1,length(latent_dims));
for latent_dim=1:16
    lat_corr_chs = squeeze(corr_chs(:,latent_dim,:));
    latentdim_mean_corrs(latent_dim) = mean(mean(lat_corr_chs));
end
[best_latentdim_mean_corr,best_latent_dim] = max(latentdim_mean_corrs);
best_latent_dim
best_latentdim_mean_corr


%mean(corr_chs)