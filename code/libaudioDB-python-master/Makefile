EXECUTABLE=audioDB
SOVERSION=0
MINORVERSION=0
LIBRARY=lib$(EXECUTABLE).so.$(SOVERSION).$(MINORVERSION)

ifeq ($(shell uname),Darwin)
override LIBRARY=lib$(EXECUTABLE).$(SOVERSION).$(MINORVERSION).dylib
endif

all:
	python setup.py build

test: ../../$(LIBRARY) all
	(cd tests && \
	 env PYTHONPATH=$$(python -c 'import distutils; import distutils.util; import sys; print "../build/lib.%s-%s" % (distutils.util.get_platform(), sys.version[0:3])') \
		LD_LIBRARY_PATH=../../.. \
		python InitialisationRelated.py)

clean:
	rm -rf tests/test* pyadb.pyc build
