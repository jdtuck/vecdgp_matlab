function obj = vdgp_fit(x,y,nmcmc,burn,thin,m,nugget,layers)
arguments
    x
    y
    nmcmc = 2000
    burn  = 1000
    thin = 2
    m = 10
    nugget = 1e-6
    layers = 2
end

if layers == 1
    model = fit_one_layer(x, y, 'nmcmc', nmcmc, 'm', m, ...
        'true_g', nugget, 'verb', 500);
elseif layers == 2
    model = fit_two_layer(x, y, 'nmcmc', nmcmc, 'm', m, ...
        'true_g', nugget, 'verb', 500);
else
    model = fit_three_layer(x, y, 'nmcmc', nmcmc, 'm', m, ...
        'true_g', nugget, 'verb', 500);
end

model = dgp_trim(model, burn, thin);
obj = vdgp_model(model);
