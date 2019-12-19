rbmfile = load('rbm_reconstruction_corr_Rvalue.mat');
rbm_corr_chs = rbmfile.corr_chs;
icafile = load('ica_reconstruction_corr_Rvalue.mat');
ica_corr_chs = icafile.corr_chs;
aefile = load('ae_reconstruction_corr_Rvalue.mat');
ae_corr_chs = aefile.corr_chs;
vaefile = load('vae_reconstruction_corr_Rvalue.mat');
vae_corr_chs = vaefile.corr_chs;
pcafile = load('pca_reconstruction_corr_Rvalue.mat');
pca_corr_chs = pcafile.corr_chs;
latentdim_num=16;
method_num =5;
mean_latentdim_corrs = zeros(method_num,latentdim_num);

startdim=2;
%rbm

for latent_dim=1:latentdim_num
    rbm_latdim_corr_chs = squeeze(rbm_corr_chs(:,latent_dim,:));
    mean_latentdim_corrs(1,latent_dim) = mean(mean(rbm_latdim_corr_chs));
end
for latent_dim=1:latentdim_num
    ica_latdim_corr_chs = squeeze(ica_corr_chs(:,latent_dim,:));
    mean_latentdim_corrs(2,latent_dim) = mean(mean(ica_latdim_corr_chs));
end
for latent_dim=1:latentdim_num
    ae_latdim_corr_chs = squeeze(ae_corr_chs(latent_dim,:,:));
    mean_latentdim_corrs(3,latent_dim) = mean(mean(ae_latdim_corr_chs));
end
for latent_dim=1:latentdim_num
    vae_latdim_corr_chs = squeeze(vae_corr_chs(latent_dim,:,:));
    mean_latentdim_corrs(4,latent_dim) = mean(mean(vae_latdim_corr_chs));
end
for latent_dim=1:latentdim_num
    pca_latdim_corr_chs = squeeze(pca_corr_chs(:,latent_dim,:));
    mean_latentdim_corrs(5,latent_dim) = mean(mean(pca_latdim_corr_chs));
end

x=startdim:1:16;
y1=mean_latentdim_corrs(1,startdim:latentdim_num);
y2=mean_latentdim_corrs(3,startdim:latentdim_num);
y3=mean_latentdim_corrs(4,startdim:latentdim_num);
y4=mean_latentdim_corrs(5,startdim:latentdim_num);
y5=mean_latentdim_corrs(6,startdim:latentdim_num);

plot(x,y1,'-r*');hold on
plot(x,y2,'-b>');hold on
plot(x,y3,'-ko');hold on
plot(x,y4,'-mp');hold on
plot(x,y5,'-gs');hold on

xlim([1,17]);
%设置中间间隔的刻度，修改1即可
set( gca, 'xtick', [1:1:17]);

xlabel('Number of latent factors');
ylabel('R-value');
title('Mean Correlation between Orignial and Reconstructed Channel Signals');

grid on;
%legend('RBM-BP','RBM-CD','ICA','AE','VAE',4);
legend('RBM','ICA','AE','VAE','PCA',4);