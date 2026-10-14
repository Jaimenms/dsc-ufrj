function h = gera_figura(nome,pos)
% Função auxiliar que gera figura principal

if nargin==0
    nome = '';
end
if nargin<2
    pos = [];
end

fig_= zeros(10,4);
y = .45;                        %   screen dependent constant
fig_(1,:) = [0,  y, .5,   y];   %   upper left
fig_(2,:) = [.5, y, .5,   y];   %   upper right
fig_(3,:) = [0,  0, .5,   y];   %   lower left
fig_(4,:) = [.5, 0, .5,   y];   %   lower right
fig_(5,:) = [0,  0.06,  1, 2*y];   %   full screen
fig_(6,:) = [0,  0, .5, 2*y];   %   left  half screen
fig_(7,:) = [.5, 0, .5, 2*y];   %   right half screen
fig_(8,:) = [0,  y,  1,   y];   %   upper half screen
fig_(9,:) = [0,  0,  1,   y];   %   lower half screen

if length(pos)==1
    pos = fig_(pos,:);
end

if isempty(findall(0,'name',nome))
    units = get(0,'Units');
    h = figure('name',nome,'numbertitle','off',...
    'color',[1 1 1],'menubar','none','Units','normalized',...
    'toolbar','none');
    if ~isempty(pos)
        set(h,'Position',pos)
    end
    set(h,'Units',units)
end