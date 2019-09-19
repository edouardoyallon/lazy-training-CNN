# Reproduce experiments to demonstrate an effective linearization as alpha grows

for LR in 1.0 0.1 0.01 0.001
do
for ALPHA in 10000000.0 1000000.0 100000.0 10000.0 1000.0 100.0 10.0 5.0 1.0 0.5 0.1 0.01
do
python train.py --scaling_factor $ALPHA  --lr $lr --gain 1.0 --schedule 'b' --loss 'mse' --length 100 --precision 'double'
done
done

# Obtain the SVD of the tangent kernel for cifar and random features

python extract_kernel.py --bs 9 --data 'random' --subset 495
python extract_kernel.py --bs 9 --subset 495
