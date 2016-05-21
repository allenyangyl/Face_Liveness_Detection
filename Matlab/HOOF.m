function Feature = HOOF(OF, bins, blocks)
Feature = zeros(1, bins * blocks * blocks);
[h, w] = size(OF);
OF(isnan(OF)) = 0;
for iBlock = 1 : blocks
    for jBlock = 1 : blocks
        Feature(((iBlock - 1) * blocks + jBlock - 1) * bins + 1 : ((iBlock - 1) * blocks + jBlock - 1) * bins + bins) = ...
            gradientHistogram(...
            real(OF(round(1+h*(iBlock-1)/(blocks+1)):round(h*(iBlock+1)/(blocks+1)), round(1+w*(jBlock-1)/(blocks+1)):round(w*(jBlock+1)/(blocks+1)))), ...
            imag(OF(round(1+h*(iBlock-1)/(blocks+1)):round(h*(iBlock+1)/(blocks+1)), round(1+w*(jBlock-1)/(blocks+1)):round(w*(jBlock+1)/(blocks+1)))), ...
            bins)';
    end
end
Feature(isnan(Feature)) = 0;
end