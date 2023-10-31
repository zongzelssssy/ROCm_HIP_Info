Ubuntu 20.04 with update software

sudo apt-get update
sudo apt-get install build-essential clang llvm-dev git

Use `llvm-config --version` see whether is 10.0.0

Install cmake 3.26.4 manually.

Install cuda(11.0) according to ![image-20230612174003287](/Users/zongzel/Library/Application Support/typora-user-images/image-20230612174003287.png)



Command is

 wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pinsudo 

mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

~~sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub~~

(Write this according to official website will get error :

**The following signatures couldn't be verified because the public key is not available : NO_PUBKEY A4B469963BF863CC**

So replace it : 

sudo apt-key del 7fa2af80
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub)

sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

sudo apt-get update

sudo apt-get -y install cuda

