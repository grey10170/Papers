mkdir -p Data/TinyImageNet/test/
pip3 install gdown
gdown --id 1xLg4NDOYvySXyi1XwIcwf1sTEyHOhQ5f --output Data/TinyImageNet.tar.gz
gdown --id 1wU6YCmkIcUXoPq7x_e7-IUhuCITOkKtM --output Data/test_lmdb.zip
gdown --id 1Tm-CMt5oaUXGSkiaF5Y0M_4PhIeBdAxl --output Ref/model_best.pth.tar
tar -zxvf Data/TinyImageNet.tar.gz
tar -xvf Data/TinyImageNet/test.tar -C Data/
mv Data/test/*/* Data/TinyImageNet/test/
rm -rf Data/test/
unzip Data/test_lmdb.zip -d Data/
python3 Ref/lsun.py export Data/test_lmdb --out_dir Data/LSUN/test/ --flat
rm -rf Data/test_lmdb/ Data/*.tar.gz Data/*.zip
