for data in "TinyImageNet" "LSUN"
do
    for transform in "Crop" "Resize"
    do
        python3 eval.py --batch_size 32 --temp 1 --epsilon 0 --dataset $data --transform $transform
        python3 eval.py --batch_size 16 --temp 1 --epsilon 0.002 --dataset $data --transform $transform
        python3 eval.py --batch_size 32 --temp 1000 --epsilon 0 --dataset $data --transform $transform
        python3 eval.py --batch_size 16 --temp 1000 --epsilon 0.002 --dataset $data --transform $transform
    done
done