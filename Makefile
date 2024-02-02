
######################################

QUIET := @
ECHO  := echo
RM    := rm

CC    := clang++

######################################

PYTHONDIR := "/home/veronesi/miniconda3"
PYVERSION := "python3.11"

TOPDIR := $(shell pwd)
SRCDIR := ${TOPDIR}/src

OUTDIR ?= ${TOPDIR}

######################################

SRCS := ${SRCDIR}/python_wrapper.cpp

INCS := \
	-I${SRCDIR}/common \
        -I${PYTHONDIR}/include/${PYVERSION} \
        -I${PYTHONDIR}/lib/${PYVERSION}/site-packages/numpy/core/include/numpy

CFLAGS  := -Wall -std=c++20 -fPIC -ferror-limit=10
LDFLAGS := -lm

######################################

PY_CFLAGS := -shared ${FLAGS}

# OUTPUT FILE NAME
PY_TARGET := MyModule.so

######################################
## TARGETS

.PHONY: python clean

python:
	${CC} ${PY_CFLAGS} ${INCS} ${LDFLAGS} -o ${OUTDIR}/${PY_TARGET} ${SRCS}

clean: 
	${QUIET}${RM} -rf ${OUTDIR}/${PY_TARGET}
	${QUIET}${RM} -f *.log *.txt *.so

