function status = odeMenuBar(t,y,flag,varargin)
% Função que possibilita acompanhar e interromper a integração de ODEs.

persistent TARGET_FIGURE TMIN TMAX


switch(flag)
    case 'init'                           % odeplot(tspan,y0,'init')        
    TMIN = t(1);
    TMAX = t(2);
    TARGET_FIGURE =progressbar2(t(1)./(TMAX-TMIN), 0)


    case 'done'                           % odeplot([],[],'done')
    TARGET_FIGURE = progressbar(TMAX(1)./(TMAX-TMIN), 0);
    if ishandle(TARGET_FIGURE)
        close(TARGET_FIGURE)
    end
end