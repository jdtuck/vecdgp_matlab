function obj = vdgp_fit(x,y,nmcmc,burn,thin,m,nugget)
arguments
    x
    y
    nmcmc = 2000
    burn  = 1000
    thin = 2
    m = 10
    nugget = 1e-6
end
model = fit_two_layer(x, y, 'nmcmc', nmcmc, 'm', m, ...
    'true_g', nugget, 'verb', 500);
model = dgp_trim(model, burn, thin);
obj = vdgp_model(model);
