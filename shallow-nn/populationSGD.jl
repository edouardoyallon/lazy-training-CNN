d = 100 # dimension of the supervised learning problem (our d-1)
m0 = 3 # number of neurons of generating data
niter = 20000 # put 20000
@assert niter>1999
m = 50

#scales = 10 .^ (-2.5:0.1:1)
scales = cat([0.01,0.02,0.04],10 .^ (-1.0:0.1:0),[2,4,8],dims=1)
nscales = length(scales)
ntrials = 10

Fs = zeros(ntrials,nscales)

batchsize = 200
stepsize = 10

p = Progress(ntrials*nscales)
Random.seed!(1)
for i = 1:ntrials
    θ0 = randn(m0,d) # random ground truth
    θ0 = θ0 ./ sqrt.(sum(θ0.^2,dims=2))
    w0  = sign.(randn(m0))

    for j=1:nscales
        scale = scales[j]
        stepsize = min(0.25/scale^2,25)
        ws,θs,val = populationSGDfor2NN(m,w0,θ0,stepsize,batchsize,scale,niter)
        Fs[i,j]= sum(val[end-1999:end])
        ProgressMeter.next!(p)
    end
end

figure(figsize=[4,4])
mea = sum(Fs,dims=1)'/ntrials
stdr = sqrt.(sum((Fs' .- mea).^2, dims=2)/(ntrials-1))
ss = 1/maximum(mea)
semilogx(scales,ss*mea,"k",linewidth=2)
fill_between(scales,ss*(mea+stdr)[:],ss*(mea-stdr)[:],color=[0.85,0.85,0.85])
ylabel("Population loss at convergence")
xlabel(L"\tau")

#vlines([0.15; 0.5],[0 ;0],[4; 4],linestyle=":")
#fill_betweenx([0; 4],[0.15 ;0.15],[0.5; 0.5],hatch="//",facecolor="None",edgecolor="k",linestyle=":",label="not yet converged")
#legend(loc="upper left")
#savefig("lazySGD_tau_sans.pdf",bbox_inches="tight")