
aefile = load('ae_reconstruction_corr_Rvalue.mat');
ae_corr_chs = aefile.corr_chs;

vaefile = load('vae_reconstruction_corr_Rvalue.mat');
vae_corr_chs = vaefile.corr_chs;

latentdim_num=16;
sub_num =32;

corr_chs = ae_corr_chs;
%corr_chs = vae_corr_chs;
mean_latentdim_corrs = zeros(sub_num,latentdim_num);

for subNo = 1:32
    for latent_dim=1:latentdim_num
        latdim_corr_chs = squeeze(corr_chs(latent_dim,subNo,:)); %获取到该sub该latdim下各个channel的重构表现
        mean_latentdim_corrs(subNo,latent_dim) = mean(latdim_corr_chs);
    end
end


startdim=2;
x=startdim:1:16;
for subNo = 1:32
    y = mean_latentdim_corrs(subNo,startdim:latentdim_num);
    plot(x,y,'-*','Color',[rand rand rand]);hold on
end

xlim([1,17]);
%设置中间间隔的刻度，修改1即可
set( gca, 'xtick', [1:1:17]);

xlabel('Number of latent factors');
ylabel('R-value');
title('Mean Correlation between Orignial and Reconstructed Channel Signals: AE based Method');

grid on;
%legend('RBM-BP','RBM-CD','ICA','AE','VAE',4);
legend({'sub01','sub02','sub03','sub04','sub05','sub06','sub07','sub08','sub09','sub10',...
'sub11','sub12','sub13','sub14','sub15','sub16','sub17','sub18','sub19','sub20',...
'sub21','sub22','sub23','sub24','sub25','sub26','sub27','sub28','sub29','sub30',...
'sub31','sub32'},4);
