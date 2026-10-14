function limpa_objetos(nomes)
% Função auxiliar que limpa objetos de uma figura

for i=1:length(nomes)
    h_obj = findall(gcf,'userdata',nomes{i});
    delete(h_obj)
end

