function colormap_pattern
% Função que permite plotar as figuras para IPE

CMAP = [0 0.7 0.25;0 0.3 0.6;0 0.3 0.9;0.5 0.5 0.5];
axis tight
set(gca,'View', [90 -90],'CLim',[1 4])
colormap(CMAP)
a=colorbar;
set(a,'YLim',[1 4],'YTickLabel',['             ';'Anular       ';'Bolhas       ';'             ';'Intermitente ';'Estratificado'])
shading interp