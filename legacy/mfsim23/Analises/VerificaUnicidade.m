% Rotina que verifica unidadede

n = 0.2;
m = 0.2;
np = 1e3;


% Variar hd
hD     = linspace(0.01,0.99,np);

hD(1)=[];
hD(end)=[];

% Adimensionais
At  = pi./4;
ALt = (pi - acos(2.*hD-1) + (2.*hD-1).*(1-(2.*hD-1).^2).^0.5)./4;
AGt = (     acos(2.*hD-1) - (2.*hD-1).*(1-(2.*hD-1).^2).^0.5)./4;
SLt = (pi - acos(2.*hD-1));
SGt = (     acos(2.*hD-1));
SIt = (                                (1-(2.*hD-1).^2).^0.5);
uLt = At./ALt;
uGt = At./AGt;
DGt = 4.*AGt./(SGt + SIt);
DLt = 4.*ALt./(SLt      );

% Funções
func_g   = (SGt./AGt+SIt./ALt+SIt./AGt).*(uGt.*DGt).^(-m).*uGt.^2;
func_f   = (uLt.*DLt).^(-n).*uLt.^2.*SLt./ALt;



X2 = 1:1:250;
Y = 1000.*ones(size(X2));

%Y = 500:10:1000;
%X2 = 160.*ones(size(Y));

figure
for i=1:length(X2)
    X2i = X2(i);
    Yi=Y(i);

    func = X2i.*func_f - func_g -4.*Yi;
    func10 = sign(func).*log10(abs(X2i.*func_f - func_g -4.*Yi));

    n = 10;
    P = polyfit(hD, func10, n);

    est = polyval(P,hD);

    plot(hD,func10)
    hold on
%    plot(hD,est,'-')
end