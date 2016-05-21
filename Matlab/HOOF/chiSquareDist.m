function distance = chiSquareDist(A)

% Caclulate Chi Squared for a vector of Histograms


L = A;
t = size(L,2);

distance = zeros(t,t);

for i=1:t
   temp = repmat(L(:,i),1,t);
   G = (temp+L>0);
   distance(i,:) = sum(G.*[(temp - L).^2]./ (G.*[temp+L]+not(G)),1) ;
end