
d = 100 # dimension of the supervised learning problem
n_train  = 1000 # size of train set
n_test   = 1000 # size of test set
m0 = 3 # nb of neurons teacher
m = 50 # nb of neurons student


scaling = 1 # we change the initialization instead of the scaling (it is equivalent, up to a square)
niter = 10000
scales  = 10 .^ (-2.2:0.1:1) # scales of init
ntrials = 10 # repetition with different random data/teacher/init/
ltrains = zeros(niter,length(scales),ntrials)
ltests = zeros(niter,length(scales),ntrials)
test_err_tangent = zeros(niter,ntrials)

p = Progress(length(scales)*ntrials) # progress bar
for k = 1:ntrials
    # random teacher
    w1 = randn(m0,d)
    w1 = w1 ./ sqrt.(sum(w1.^2, dims=2))
    w2  = sign.(randn(m0))
    f(X) = sum( w2 .* max.( w1 * X', 0.0), dims=1)

    # data sets
    X_train = randn(n_train, d)
    X_train = X_train  ./ sqrt.(sum(X_train.^2, dims=2))
    Y_train = f(X_train) #randn(1,n_train)
    X_test  = randn(n_test, d)
    X_test  = X_test ./ sqrt.(sum(X_test.^2, dims=2))
    Y_test  = f(X_test);

    # initialization
    W_init = randn(m, d+1)
    # symmetrization
    W_init[1:div(m,2),end] = abs.(W_init[1:div(m,2),end])
    W_init[(div(m,2)+1):end,end] = - W_init[1:div(m,2),end]
    W_init[(div(m,2)+1):end,1:end-1] = W_init[1:div(m,2),1:end-1]
    W_init0 = W_init;

    for i=1:length(scales)
        W_init = scales[i]*W_init0 # both layers are multiplied so scale ~ alpha^2
        # the linear scaling of the step-size works for large scales only
        stepsize = min(10,0.1/scales[i].^2)
        Ws, loss_train, loss_test = GDfor2NN(X_train, X_test, Y_train, Y_test, W_init, scaling, stepsize, niter)
        ltrains[:,i,k] = loss_train
        ltests[:,i,k] = loss_test
        ProgressMeter.next!(p)
    end
end

# Compute mean and std
meana = sum(ltests[end,:,:],dims=2)/ntrials
meanb = sum(minimum(ltests,dims=1),dims=3)[:]/ntrials
stda = sqrt.(sum((ltests[end,:,:] .- meana).^2,dims=2)/(ntrials-1))
stdb = sqrt.(sum((minimum(ltests,dims=1) .- meanb').^2,dims=3)[:]/(ntrials-1))

# Plot
figure(figsize=[4,4])
ss = 1000 # for nicer yticks
fill_between(scales, ss*(meana+stda)[:],ss*(meana-stda)'[:],color=[0.85,0.85,0.85])
fill_between(scales, ss*(meanb+stdb)[:],ss*(meanb-stdb)'[:],color=[0.85,0.85,0.85])
semilogx(scales, ss*meana,"k",alpha=1,linewidth=3,label="end of training")
semilogx(scales, ss*sum(minimum(ltests,dims=1),dims=3)[:]/ntrials,":k",alpha=1,linewidth=3,label="best throughout training")
ylabel("Test loss")
xlabel(L"\tau")
legend()
#savefig("test_loss_tau.pdf",bbox_inches="tight")