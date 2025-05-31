# Simple Makefile for micrograd-c

CC = gcc
CFLAGS = -c -O2 -Wall -Werror -std=c99

all: value.dll mlp.dll

value.o: value.c value.h
	$(CC) $(CFLAGS) -o value.o value.c

mlp.o: mlp.c mlp.h value.h
	$(CC) $(CFLAGS) -o mlp.o mlp.c

value.dll: value.o
	$(CC) -shared -o value.dll value.o -Wl,--out-implib,libvalue.dll.a

mlp.dll: mlp.o value.o
	$(CC) -shared -o mlp.dll mlp.o value.o -Wl,--out-implib,libmlp.dll.a

install: value.dll mlp.dll
	copy value.dll ..\micrograd_c\lib\
	copy mlp.dll ..\micrograd_c\lib\

clean:
	del /Q *.o *.dll *.dll.a

.PHONY: all clean install
