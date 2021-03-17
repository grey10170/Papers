for useLRN in False True:
do
    for useDropOut in False True:
    do
        for useAug in False True:
        do
            python3 train.py  --useLRN $useLRN --useDropOut $useDropOut --useAug $useAug
        done
    done
done
