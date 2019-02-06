CC=nvcc
# this is for K80
FLAG = -arch sm_37
# this is for v100
#FLAG = -arch sm_70
src = $(wildcard *.cu)
exe = $(src:.cu=.exe)



all: $(exe)

%.exe : %.cu
	$(CC) $(FLAG) -o $@ $<



clean :
	rm -f *.exe



