d = 100 # dimension of the supervised learning problem
n_train  = 1000 # nb of data points
n_test   = 1000
m0 = 3 # nb of neurons of ground truth

scaling = 1
niter = 25000
ms = [2,3,4,6,8,12,16,24,32,64,128,256,512]
ntrials = 10

# compute with alpha = 1/sqrt(m)
m_ltrains = zeros(niter,length(ms),ntrials)
m_ltests = zeros(niter,length(ms),ntrials)

# compute with alpha = 1/m
m_ltrains2 = zeros(niter,length(ms),ntrials)
m_ltests2 = zeros(niter,length(ms),ntrials)

p = Progress(length(ms)*ntrials*2)
for k = 1:ntrials
    # ground thruth
    w1 = randn(m0,d)
    w1 = w1 ./ sqrt.(sum(w1.^2, dims=2))
    w2 = sign.(randn(m0))
    f(X) = sum( w2 .* max.( w1 * X', 0.0), dims=1)*100 # neurons

    # data sets
    X_train = randn(n_train, d)
    X_train = X_train  ./ sqrt.(sum(X_train.^2, dims=2))
    Y_train = f(X_train) #randn(1,n_train)
    X_test  = randn(n_test, d)
    X_test  = X_test ./ sqrt.(sum(X_test.^2, dims=2))
    Y_test  = f(X_test)
    
    # compute with alpha = 1/sqrt(m)
    for i=1:length(ms)
        m = ms[i]
        W_init = randn(m, d+1)
        scaling = 1/sqrt(m)
        stepsize = 1/m
        Ws, loss_train, loss_test = GDfor2NN(X_train, X_test, Y_train, Y_test, W_init, scaling, stepsize, niter);
        m_ltrains[:,i,k] = loss_train
        m_ltests[:,i,k] = loss_test
        ProgressMeter.next!(p)
    end
    
    # compute with alpha = 1/m
    for i=1:length(ms)
        m = ms[i]
        W_init = randn(m, d+1)
        scaling = 1/m
        stepsize = 0.05/m
        Ws, loss_train, loss_test = GDfor2NN(X_train, X_test, Y_train, Y_test, W_init, scaling, stepsize, niter)
        m_ltrains2[:,i,k] = loss_train
        m_ltests2[:,i,k] = loss_test
        ProgressMeter.next!(p)
    end
end

# Prepare the plots
sa=1
sb=length(ms)
ss = .1
#endtraining = permutedims(minimum(m_ltests[:,sa:sb,:],dims=1),[2 3 1])[:,:,1]
meana = sum(m_ltests[end,sa:sb,:],dims=2)/ntrials
#meanb = sum(minimum(m_ltests[:,sa:sb,:],dims=1),dims=3)[:]/ntrials
meana2 = sum(m_ltests2[end,sa:sb,:],dims=2)/ntrials
#meanb2 = sum(minimum(m_ltests[:,sa:sb,:],dims=1),dims=3)[:]/ntrials
stda = sqrt.(sum((m_ltests[end,sa:sb,:] .- meana).^2,dims=2)/(ntrials-1))
stdb = sqrt.(sum((minimum(m_ltests[:,sa:sb,:],dims=1) .- meanb').^2,dims=3)[:]/(ntrials-1))
confint_low = sort(endtraining,dims=dims=2)[:,1]
confint_up  = sort(endtraining,dims=dims=2)[:,end]


figure(figsize=[4,4])

#fill_between(ms[sa:sb],ss*(meana+stda)[:],ss*(meana-stda)'[:],color=[0.85,0.85,0.85])
#fill_between(ms[sa:sb],ss*(meana2+stdb)[:],ss*(meanb-stdb)'[:],color=[0.85,0.85,0.85])
#fill_between(ms[sa:sb],ss*confint_low[:],ss*confint_up[:],color=[0.85,0.85,0.85])
semilogx(ms[sa:sb],ss*m_ltests[end,:,:],"ok",markersize=1);
semilogx(ms[sa:sb],ss*m_ltests2[end,:,:],"o",color=[0.5,0.5,0.5],markersize=1);

semilogx(ms[sa:sb],ss*meana,"k",alpha=1,linewidth=3,label=L"scaling $1/\sqrt{m}$")
semilogx(ms[sa:sb],ss*meana2,color=[0.5,0.5,0.5],linewidth=3,label=L"scaling $1/m$")

ylabel("Test loss")
xlabel(L"m")
xticks([1, 10, 100, 1000])
yticks([0,1])
legend()
#savefig("test_mcomp_dots.pdf",bbox_inches="tight")