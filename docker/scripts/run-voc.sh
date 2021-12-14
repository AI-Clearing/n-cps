cd scripts
export env_file="voc.env"

for r in 50 #101
do
    export resnet=$r
    # export labeled_ratio=16
    # export nepochs=32
    # for i in 2 3
    # do
    #     export num_networks=$i    
    #     ./run-single-voc.sh
    #     sleep 5
    # done

    export labeled_ratio=8
    export nepochs=34
    for i in 3 2
    do
        export num_networks=$i    
        ./run-single-voc.sh
    done

    # export labeled_ratio=4
    # export nepochs=40
    # for i in 2 3
    # do
    #     export num_networks=$i    
    #     ./run-single-voc.sh
    # done

    # export labeled_ratio=2
    # export nepochs=60
    # for i in 2 3
    # do
    #     export num_networks=$i    
    #     ./run-single-voc.sh
    # done
done

cd ..

sudo shutdown -P now