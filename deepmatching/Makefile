CC=g++

OS_NAME=$(shell uname -s)
ifeq ($(OS_NAME),Linux) 
  LAPACKLDFLAGS=/usr/lib/libatlas.so   # single-threaded blas
  #LAPACKLDFLAGS=/usr/lib64/atlas/libatlas.so  # multi-threaded blas
  #BLAS_THREADING=-D MULTITHREADED_BLAS # remove this if wrong
endif
ifeq ($(OS_NAME),Darwin)  # Mac OS X
  LAPACKLDFLAGS=-framework Accelerate # for OS X
endif

LAPACKCFLAGS=-Dinteger=int $(BLAS_THREADING)
LIB_DIR_X86=/usr/lib/x86_64-linux-gnu
STATICLAPACKLDFLAGS=-fPIC -Wall -g -fopenmp -static -static-libstdc++ $(LIB_DIR_X86)/libjpeg.a $(LIB_DIR_X86)/libpng.a $(LIB_DIR_X86)/libz.a /usr/lib/libblas.a /usr/lib/gcc/x86_64-linux-gnu/5/libgfortran.a /usr/lib/gcc/x86_64-linux-gnu/5/libquadmath.a # statically linked version

CFLAGS= -fPIC -Wall -g -std=c++11 $(LAPACKCFLAGS) -fopenmp -DUSE_OPENMP -O3
LDFLAGS=-fPIC -Wall -g -ljpeg -lpng -fopenmp 

# configure CpythonFlag
# CPYTHONFLAGS=-I/usr/local/lib/python2.7 -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include
PYTHONLIB = $(shell python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
NUMPYLIB = $(shell python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())+'/numpy/core/include'")
CPYTHONFLAGS=-I$(PYTHONLIB) -I$(NUMPYLIB) -I/usr/include/python2.7

LIBFLAGS= -L/lib/x86_64-linux-gnu -lpng -ljpeg -lz -lblas
SOURCES := $(shell find . -name '*.cpp' ! -name 'deepmatching_matlab.cpp')
OBJ := $(SOURCES:%.cpp=%.o)
HEADERS := $(shell find . -name '*.h')


all: deepmatching-static
.cpp.o:  %.cpp %.h
	$(CC) -o $@ $(CFLAGS) -c $+

deepmatching: $(HEADERS) $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS)

deepmatching-static: $(HEADERS) $(OBJ)
	$(CC) -o $@ $^ $(STATICLAPACKLDFLAGS)

python: $(HEADERS) $(OBJ)
#	swig -python $(CPYTHONFLAGS) deepmatching.i # not necessary, only do if you have swig compiler
	g++ $(CFLAGS) -c deepmatching_wrap.c $(CPYTHONFLAGS)
	g++ -shared $(LDFLAGS) $(LAPACKLDFLAGS) deepmatching_wrap.o $(OBJ) -o _deepmatching.so $(LIBFLAGS) 

clean:
	rm -f $(OBJ) deepmatching *~ *.pyc .gdb_history deepmatching_wrap.o _deepmatching.so deepmatching.mex???
