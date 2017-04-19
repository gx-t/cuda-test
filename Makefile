all: 00-rel

00-deb:
	nvcc -g 00-config.cu -o 00-config
00-rel:
	nvcc -O2 00-config.cu -o 00-config
	strip 00-config

ctags:
	ctags -R . /usr/local/cuda/include

