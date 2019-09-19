# generate the data
d = 2 # dimension of input

# random teacher 2-NN
m0 = 3 # nb of neurons teacher
w1 = randn(m0,d)
w1 = w1 ./ sqrt.(sum(w1.^2, dims=2))
w2  = sign.(randn(m0))
f(X) = sum( w2 .* max.( w1 * X', 0.0), dims=1)

# data sets
n_train  = 15 # size train set (15)
n_test   = 20  # size test set
X_train = randn(n_train, d)
X_train = X_train  ./ sqrt.(sum(X_train.^2, dims=2))
Y_train = f(X_train)
X_test  = randn(n_test, d)
X_test  = X_test ./ sqrt.(sum(X_test.^2, dims=2))
Y_test  = f(X_test);

# initialize and train
m = 16 # nb of neurons student
scaling = 1
niter = 10^5
stepsize = 0.005

# initialization
W_init = randn(m, 2)
W_init = W_init  ./ sqrt.(sum(W_init.^2, dims=2))
W_init = cat(W_init, rand(m),dims=2)

# symmetrization to set initial output to zero (optional)
W_init[(div(m,2)+1):end,end] = - W_init[1:div(m,2),end]
W_init[(div(m,2)+1):end,1:end-1] = W_init[1:div(m,2),1:end-1] 

# choose scale of init (0.1 not lazy / 2 lazy)
W_init = 0.2*W_init

@time Ws, loss_train, loss_test = GDfor2NN(X_train, X_test, Y_train, Y_test, W_init, scaling, stepsize, niter);


figure(figsize=[8,4])

subplot(121)
semilogy(loss_train,label="train loss")
semilogy(loss_test,label="test loss")
legend();title("Convergence");


# things to plot
iters = Int.(floor.(exp.(range(0, stop = log(niter), length = 100))))#cat(1:20,21:4:100,110:15:500,500:100:10000,20000:1000:niter,dims=1) 
mid=div(m,2)
finalsign = sign.(Ws[:,3,end])
pxs = Ws[finalsign.>0,1,iters] .* Ws[finalsign.>0,3,iters]
pys = Ws[finalsign.>0,2,iters] .* Ws[finalsign.>0,3,iters]
pxsm = Ws[finalsign.<0,1,iters] .* abs.(Ws[finalsign.<0,3,iters])
pysm = Ws[finalsign.<0,2,iters] .* abs.(Ws[finalsign.<0,3,iters])
px0 = w1[:,1] #.* w2
py0 = w1[:,2] #.* w2

subplot(122)
r = 1
plot(r*cos.(0.0:0.01:2π),r*sin.(0.0:0.01:2π),":",color="k",label="circle of radius $(r)")

arrow(0,0,px0[1],py0[1],head_width=0.06,length_includes_head=true,facecolor="C3")
arrow(0,0,px0[2],py0[2],head_width=0.06,length_includes_head=true,facecolor="C0")
arrow(0,0,px0[3],py0[3],head_width=0.06,length_includes_head=true,facecolor="C3",label="teacher")

plot(pxs',pys',linewidth=1.0,"C3");
plot(pxs[1,:],pys[1,:],linewidth=0.5,"C3",label="gradient flow (+)")
scatter(pxs[:,end],pys[:,end],30,color="C3")
plot(pxsm',pysm',linewidth=1.0,"C0");
plot(pxsm[1,:],pysm[1,:],linewidth=0.5,"C0",label="gradient flow (-)")
scatter(pxsm[:,end],pysm[:,end],30,color="C0")

bx= max(max(maximum(abs.(pxs)), maximum(abs.(pys)))*1.1,1.1)
axis([-bx,bx,-bx,bx]);
axis("off")

#legend(loc=3)
#savefig("cover_lazy_leg.pdf",bbox_inches="tight")
#savefig("gf_doubling_1.png")