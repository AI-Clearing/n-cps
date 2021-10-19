cd scripts
export env_file="city.env"

for r in 50 101
do
    export resnet=$r

    export labeled_ratio=16
    export nepochs=128
    for i in 3
    do
        export num_networks=$i    
        ./run-single-city.sh
        sleep 5
    done

    export labeled_ratio=8
    export nepochs=137
    for i in 3
    do
        export num_networks=$i    
        ./run-single-city.sh
        sleep 5
    done

    export labeled_ratio=4
    export nepochs=160
    for i in 3
    do
        export num_networks=$i    
        ./run-single-city.sh
        sleep 5
    done

    export labeled_ratio=2
    export nepochs=240
    for i in 3
    do
        export num_networks=$i    
        ./run-single-city.sh
        sleep 5
    done
done

cd ..

sudo shutdown -P now