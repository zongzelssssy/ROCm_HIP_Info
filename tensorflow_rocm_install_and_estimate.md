When finish ROCm platform setup. (rocminfo and rocm-smi output correctly.)

Tensorflow installation:
Refer from https://github.com/RadeonOpenCompute/ROCm/blob/develop/docs/how_to/tensorflow_install/tensorflow_install.md



1. Make sure your python version is 3.7/3.8/3.9/3.10
2. Setup pip command
3. Install wheel use:

 ```/usr/bin/python[version] -m pip install --user tensorflow-rocm==[wheel-version] --upgrade```

​		wheel version can be found from https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/blob/develop-upstream/rocm_docs/tensorflow-rocm-release.md

4. Setup dependency

​		This step is very tedious and need to do a lot of trivial version-change operation. Here is an axample for rocm5.4.3 and wheel tensorflow-rocm 2.11.0.540.

​		You need to keep :

```tensorboard 2.11.2 requires protobuf<4,>=3.9.2```

```tensorflow-rocm 2.11.1.550 requires numpy<1.23,>=1.20```

```tensorflow-rocm 2.11.1.550 requires protobuf<3.20,>=3.9.2```

```tensorflow 2.12.0 requires keras<2.13,>=2.12.0```

```tensorflow 2.12.0 requires numpy<1.24,>=1.22```

```tensorflow 2.12.0 requires tensorboard<2.13,>=2.12```

```tensorflow 2.12.0 requires tensorflow-estimator<2.13,>=2.12.0```

```tensorflow-metadata 1.13.1 requires protobuf<5,>=3.20.3```

......

​		It's determined by your own environment and maybe need more version-change problem. It's actually unsmart.

5. Install libraries

```sudo apt install rocm-libs rccl```

6. Install main-body of tensorflow

```sudo pip3 install tensorflow```

7. Set environment variable

```export PYTHONPATH="./.local/lib/python[version]/site-packages:$PYTHONPATH"```

8. Test installation

```python3 -c 'import tensorflow' 2> /dev/null && echo 'Success' || echo 'Failure'```

or 

```python
python
import tensorflow
```

​		This step maybe you will meet an error ```ImportError: libhsa-runtime64.so.1: cannot open shared object file: No such file or directory```

​		Try add sudo: 

```sudo python3 -c 'import tensorflow' 2> /dev/null && echo 'Success' || echo 'Failure'```



