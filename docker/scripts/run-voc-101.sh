cd scripts
export env_file="ResNet101.env"
export labeled_ratio=16
for i in 2 3
do
    export num_networks=$i    
    ./run-single.sh
    sleep 5
done

export labeled_ratio=8
export nepochs=34
for i in 2 3
do
    export num_networks=$i    
    ./run-single.sh
done

export labeled_ratio=4
export nepochs=40
for i in 2 3
do
    export num_networks=$i    
    ./run-single.sh
done

export labeled_ratio=2
export nepochs=60
for i in 2 3
do
    export num_networks=$i    
    ./run-single.sh
done

cd ..

sudo shutdown -P now