// pyadbmodule.c
// 
// the internal portion of the wrapper for audio
// see pyadb.py for the public classes
// 
// Created by Benjamin Fields on 2009-09-04.
// Big update for direct data insertion 2010-June (Benjamin Fields)
// Copyleft 2009, 2010 Goldsmith University of London. 
// Distributed and licensed under GPL2. See ../../license.txt for details.
// 	
#include <fcntl.h>
#include <string.h>
#include "Python.h"
#include "structmember.h"
#include "audioDB/audioDB_API.h"
#include "numpy/arrayobject.h"

#define ADB_HEADER_FLAG_L2NORM		(0x1U)
#define ADB_HEADER_FLAG_POWER		(0x4U)
#define ADB_HEADER_FLAG_TIMES		(0x20U)
#define ADB_HEADER_FLAG_REFERENCES	(0x40U)

static void _pyadb_close(void *ptr);

/* create a new database */
/* returns a struct or NULL on failure */
/* api call: */
/* adb_t *audiodb_create(const char *path, unsigned datasize, unsigned ntracks, unsigned datadim);*/
PyObject * _pyadb_create(PyObject *self, PyObject *args)
{
	unsigned datasize, ntracks, datadim;
	const char *path;
	int ok;
	adb_t *new_database;
	ok = PyArg_ParseTuple(args, "sIII", &path, &datasize, &ntracks, &datadim);
	if (!ok) return 0;
	new_database = audiodb_create(path, datasize, ntracks, datadim);
	if (!new_database) return 0;
	if(audiodb_l2norm(new_database)) {
          unlink(path);
          return 0;
        }
	return PyCObject_FromVoidPtr( new_database, _pyadb_close);
}

/* open an existing database */
/* returns a struct or NULL on failure */
/* flags expects fcntl flags concerning the opening mode */
/* api call: */
// adb_t *audiodb_open(const char *path, int flags);
PyObject * _pyadb_open(PyObject *self, PyObject *args)
{
	const char *path;
	char mode;
	int ok;//in python layer need to translate boolean flags to byte mask
	adb_t *fresh_database;
	ok = PyArg_ParseTuple(args, "sc", &path, &mode);
	if (!ok) return 0;
	if (mode == 'r'){
		fresh_database = audiodb_open(path, O_RDONLY);
	}else if (mode == 'w'){
		fresh_database = audiodb_open(path, O_RDWR);
	}else{
		PyErr_SetString(PyExc_ValueError,
		"mode must be either \'r\' or \'w\'.  It appears to be something else.");
		return 0;
	}
	if (!fresh_database) return 0;
	
	return PyCObject_FromVoidPtr( fresh_database, _pyadb_close);	
}

/* database status */  
/* api call: */
// int audiodb_status(adb_t *mydb, adb_status_ptr status);
PyObject * _pyadb_status(PyObject *self, PyObject *args)
{
	adb_t *check_db;
	adb_status_t *status;
	int flags, ok;
	PyObject * incoming = 0;
	status = (adb_status_t *)malloc(sizeof(adb_status_t));
	
	ok = PyArg_ParseTuple(args, "O", &incoming);
	if (!ok) return 0;
	check_db = (adb_t *)PyCObject_AsVoidPtr(incoming);
	
	
	flags = audiodb_status(check_db, status);
	return Py_BuildValue("IIIIILL", status->numFiles, 
									status->dim, 
									status->dudCount, 
									status->nullCount, 
									status->flags, 
									status->length, 
									status->data_region_size);

}

/*engage l2norm in the referenced db*/
/*api call:*/
//int audiodb_l2norm(adb_t *mydb);
PyObject * _pyadb_l2norm(PyObject *self, PyObject *args)
{
	adb_t *current_db;
	int ok;
	PyObject * incoming = 0;
	
	ok = PyArg_ParseTuple(args, "O", &incoming);
	if (!ok) return 0;
	current_db = (adb_t *)PyCObject_AsVoidPtr(incoming);
	
	ok = audiodb_l2norm(current_db);
	return PyBool_FromLong(ok-1);
	
}

/*engage power thresholding in the referenced db*/
/*api call:*/
// int audiodb_power(adb_t *mydb);
PyObject * _pyadb_power(PyObject *self, PyObject *args)
{
	adb_t *current_db;
	int ok;
	PyObject * incoming = 0;
	
	ok = PyArg_ParseTuple(args, "O", &incoming);
	if (!ok) return 0;
	current_db = (adb_t *)PyCObject_AsVoidPtr(incoming);
	
	ok = audiodb_power(current_db);
	return PyBool_FromLong(ok-1);
	
}

/* insert feature data from a numpy array */
/* array given should have ndarray.shape = (numVectors, numDims)*/
/* array datatype needs to be doubles (float may work...)*/
/* if power is given, must be 1d array of length numVectors*/
/* if times is given, must be 1d array of length 2*numVectors like this:*/

/* api call: */
// typedef struct adb_datum {
//   uint32_t nvectors;
//   uint32_t dim;
//   const char *key;
//   double *data;
//   double *power;
//   double *times;
// } adb_datum_t;
// int audiodb_insert_datum(adb_t *, const adb_datum_t *);
PyObject * _pyadb_insertFromArray(PyObject *self, PyObject *args, PyObject *keywds)
{
	adb_t *current_db;
	adb_status_t *status;
	adb_datum_t *ins;
	int ok;
	npy_intp dims[1];
	unsigned int nDims = 0;
	unsigned int nVect = 0;
	PyObject *incoming = 0;
	PyArrayObject *features = 0;
	PyArrayObject *power = 0;
	PyArrayObject *times = 0;
	const char *key = NULL;
	PyArray_Descr *descr;
	static char *kwlist[]  = { "db", "features", "nDim", "nVect", "power", "key", "times" , NULL};
	
	ok =  PyArg_ParseTupleAndKeywords(args, keywds, "OO!II|O!sO!", kwlist, &incoming, &PyArray_Type, &features, &nDims, &nVect, &PyArray_Type,  &power, &key, &PyArray_Type, &times);
	if (!ok){
	  PyErr_SetString(PyExc_TypeError, "Failed at PyArg_ParseTupleAndKeywords");
	  return NULL;
	}
	//check our arrays
	// if (!PyArray_Check(features)){
	// 	PyErr_SetString(PyExc_TypeError, "features must be a numpy array (of floats or doubles)");
	// 	return NULL;
	// }
	if (!PyArray_ISFLOAT(features)){
		PyErr_SetString(PyExc_TypeError, "features numpy array must contain floats or doubles");
		return NULL;
	}
	if ((PyArray_NDIM(features) != 1) || (PyArray_DIMS(features)[0] != (nDims * nVect))){
		PyErr_SetString(PyExc_TypeError, "features numpy array must be flattened before call.");
		return NULL;
	}
	descr = PyArray_DescrFromType(NPY_DOUBLE);
	
	if (power){
		if (!PyArray_Check(power)){
			PyErr_SetString(PyExc_TypeError, "power, if given, must be a numpy array (of floats or doubles)");
			return NULL;
		}
		if (!PyArray_ISFLOAT(power)){
			PyErr_SetString(PyExc_TypeError, "power numpy array, if given, must contain floats or doubles");
			return NULL;
		}
		// power = (PyArrayObject *)PyCObject_AsVoidPtr(incomingPow);
		if (PyArray_NDIM(features) != 1 || PyArray_DIMS(power)[0] != nVect){
			PyErr_SetString(PyExc_ValueError, "power, if given must be a 1d numpy array with shape =  (numVectors,)");
			return NULL;
		}
	}
	if (times){
		if (!PyArray_Check(times)){
			PyErr_SetString(PyExc_TypeError, "times, if given, must be a numpy array (of floats or doubles)");
			return NULL;
		}
		if (!PyArray_ISFLOAT(times)){
			PyErr_SetString(PyExc_TypeError, "times numpy array, if given, must contain floats or doubles");
			return NULL;
		}
		// times = (PyArrayObject *)PyCObject_AsVoidPtr(incomingTime);
		if (PyArray_NDIM(times) != 1 || PyArray_DIMS(times)[0] != (nVect*2)){
			PyErr_SetString(PyExc_ValueError, "times, if given must be a 1d numpy array with shape =  (numVectors*2,)");
			return NULL;
		}
	}
	current_db = (adb_t *)PyCObject_AsVoidPtr(incoming);
	status = (adb_status_t *)malloc(sizeof(adb_status_t));
	//verify that the data to be inserted is the correct size for the database.
	
	ins = (adb_datum_t *)malloc(sizeof(adb_datum_t));
	if (PyArray_AsCArray((PyObject**)&features, &(ins->data), dims,  1, descr)){
		PyErr_SetString(PyExc_RuntimeError, "Trouble expressing the feature np array as a C array.");
		return NULL;
	}
	
	if (power){
	  if (PyArray_AsCArray((PyObject**)&power, &(ins->power), dims,  1, descr)){
			PyErr_SetString(PyExc_RuntimeError, "Trouble expressing the power np array as a C array.");
			return NULL;
		}
	}else{
		ins->power=NULL;
	}
	
	if (times){
	  if (PyArray_AsCArray((PyObject**)&times, &(ins->times), dims,  1, descr)){
			PyErr_SetString(PyExc_RuntimeError, "Trouble expressing the times np array as a C array.");
			return NULL;
		}
	}else{
		ins->times=NULL;
	}
	ins->key = key;
	ins->nvectors = (uint32_t)nVect;
	ins->dim = (uint32_t)nDims;
	//printf("features::%s\npower::%s\nkey::%s\ntimes::%s\n", ins->features, ins->power, ins->key, ins->times);
	ok = audiodb_insert_datum(current_db, ins);//(current_db, ins);
	return PyInt_FromLong(ok);
	
}

/* insert feature data stored in a file */
/* this is a bit gross, */
/* should be replaced eventually by a numpy based feature.*/
/* api call: */
// struct adb_insert {
//   const char *features;
//   const char *power;
//   const char *key;
//   const char *times;
// };
// int audiodb_insert(adb_t *mydb, adb_insert_t *ins);
PyObject * _pyadb_insertFromFile(PyObject *self, PyObject *args, PyObject *keywds)
{
	adb_t *current_db;
	adb_insert_t *ins;
	int ok;
	const char *features;
	const char *power = NULL;
	const char *key = NULL;
	const char *times = NULL;
	PyObject * incoming = 0;
	static char *kwlist[]  = { "db", "features", "power", "key", "times" , NULL};
	
	ok =  PyArg_ParseTupleAndKeywords(args, keywds, "Os|sss", kwlist, &incoming, &features, &power, &key, &times);
	if (!ok){return NULL;}
	
	current_db = (adb_t *)PyCObject_AsVoidPtr(incoming);
	ins = (adb_insert_t *)malloc(sizeof(adb_insert_t));
	ins->features = features;
	ins->power = power;
	ins->key = key;
	ins->times = times;
	//printf("features::%s\npower::%s\nkey::%s\ntimes::%s\n", ins->features, ins->power, ins->key, ins->times);
	ok = audiodb_insert(current_db, ins);
	return PyBool_FromLong(ok-1);
	
}

/* liszt - list strings, sizes, and time-points of all database entries
 *
 */
PyObject* _pyadb_liszt(PyObject *self, PyObject *args)
{
  adb_t *current_db;
  int ok,i;
  PyObject * incoming = NULL;
  PyObject * outgoing = NULL;
  PyObject * newBits = NULL;  

  ok = PyArg_ParseTuple(args, "O", &incoming);
  
  if (!ok) return 0;
  current_db = (adb_t *)PyCObject_AsVoidPtr(incoming);
  
  adb_liszt_results_t *liszt = audiodb_liszt(current_db);

  outgoing  = PyList_New((Py_ssize_t)0);
  for (i=0 ; i<liszt->nresults ; i++){
    newBits = Py_BuildValue("sI",liszt->entries[i].key,liszt->entries[i].nvectors);
    if (PyList_Append(outgoing,  newBits)){
      //error msg here
      Py_XDECREF(newBits);
      return NULL;
    }
    Py_DECREF(newBits);
  }
  audiodb_liszt_free_results(current_db, liszt);
  return outgoing;
}

/* base query.  The nomenclature here is about a far away as pythonic as is possible. 
 * This should be taken care of via the higher level python structure
 * returns a dict that should be result ordered and key = result key
 * and value is a list of tuples one per result associated with that key, of the form:
 *   (dist, qpos, ipos)
 * Note as well that this is by no means the most efficient way to cast from C, simply the most direct
 * and what it lacks in effeciency it gains in python side access.  It remains to be seen if this is 
 * a sensible trade.
 * api call:
 * adb_query_results_t *audiodb_query_spec(adb_t *, const adb_query_spec_t *);
 ***/
PyObject * _pyadb_queryFromKey(PyObject *self, PyObject *args, PyObject *keywds)
{
	adb_t *current_db;
	adb_query_spec_t *spec;
	adb_query_results_t *result;
	int ok, exhaustive, falsePositives;
	uint32_t i;
	const char *key;
	const char *accuMode = "db";
	const char *distMode = "dot";
	const char *resFmt = "dict";
	uint32_t hop = 0;
	double radius = 0;
	double absThres = 0; 
	double relThres = 0; 
	double durRatio = 0;
	PyObject *includeKeys = NULL;
	PyObject *excludeKeys = NULL;
	PyObject *incoming = 0;
	PyObject *outgoing = NULL;
	PyObject *thisKey = NULL;
	PyObject *currentValue = 0;
	PyObject *newBits = 0;
	static char *kwlist[]  = { "db", "key", 
								"seqLength", 
								"seqStart", 
								"exhaustive", 
								"falsePositives",
								"accumulation",
								"distance",
								"npoints",//nearest neighbor points per track
								"ntracks",
								"includeKeys",
								"excludeKeys",
								"radius",
								"absThres",
								"relThres",
								"durRatio",
								"hopSize",
                                "resFmt",
                                NULL
								};
	spec = (adb_query_spec_t *)malloc(sizeof(adb_query_spec_t));
	spec->qid.datum = (adb_datum_t *)malloc(sizeof(adb_datum_t));
	result = (adb_query_results_t *)malloc(sizeof(adb_query_results_t));
	
	spec->qid.sequence_length = 16;
	spec->qid.sequence_start = 0;
	spec->qid.flags = 0;
	spec->params.npoints = 1;
	spec->params.ntracks = 100;//number of results returned in db mode
	spec->refine.flags = 0;
	
	ok =  PyArg_ParseTupleAndKeywords(args, keywds, "Os|iiiissIIOOddddIs", kwlist, 
												&incoming, &key, 
												&spec->qid.sequence_length, 
												&spec->qid.sequence_start, 
												&exhaustive, &falsePositives,
												&accuMode,&distMode,
												&spec->params.npoints,
												&spec->params.ntracks,
												&includeKeys, &excludeKeys,
												&radius, &absThres, &relThres, &durRatio, &hop,
												&resFmt
												);
	
	if (!ok) {return NULL;}
	current_db = (adb_t *)PyCObject_AsVoidPtr(incoming);
	
	if (exhaustive){
		spec->qid.flags = spec->qid.flags | ADB_QID_FLAG_EXHAUSTIVE;
	}
	if (falsePositives){
		spec->qid.flags = spec->qid.flags | ADB_QID_FLAG_ALLOW_FALSE_POSITIVES;
	}
	
	//set up spec->params
	if (strcmp(accuMode,"db")==0){
		spec->params.accumulation = ADB_ACCUMULATION_DB;
	} else if (strcmp(accuMode,"track")==0){
		spec->params.accumulation = ADB_ACCUMULATION_PER_TRACK;
	} else if (strcmp(accuMode,"one2one")==0){
		spec->params.accumulation = ADB_ACCUMULATION_ONE_TO_ONE;
	} else{
		PyErr_SetString(PyExc_ValueError, 
			"Poorly specified distance mode. distance must either be \'db\', \'track\' or  \'one2one\'.\n");
		return NULL;
	}
	if (strcmp(distMode, "dot")==0){
	        spec->params.distance = ADB_DISTANCE_DOT_PRODUCT;
	}else if (strcmp(distMode, "eucNorm")==0){
	        spec->params.distance = ADB_DISTANCE_EUCLIDEAN_NORMED;
	}else if (strcmp(distMode, "euclidean")==0){
	        spec->params.distance = ADB_DISTANCE_EUCLIDEAN;
	}else if (strcmp(distMode, "kullback")==0){
	        spec->params.distance = ADB_DISTANCE_KULLBACK_LEIBLER_DIVERGENCE;
	}else{
	        PyErr_SetString(PyExc_ValueError, 
			  "Poorly specified distance mode. distance must either be \'dot\', \'eucNorm\' ,\'euclidean\' or \'kullback\'.\n");
	        return NULL;
	}
	
	//set up spec->refine
	//include/exclude keys
	if (includeKeys){
		if (!PyList_Check(includeKeys)){
			PyErr_SetString(PyExc_TypeError, "Include keys must be specified as a list of strings.\n");
			return NULL;
		}
		spec->refine.flags = spec->refine.flags | ADB_REFINE_INCLUDE_KEYLIST;
		spec->refine.include.nkeys = (uint32_t)PyList_Size(includeKeys);
		spec->refine.include.keys = (const char **)calloc(sizeof(const char *), spec->refine.include.nkeys);
		for (i=0;i<spec->refine.include.nkeys;i++){
			 if (PyString_Check(PyList_GetItem(includeKeys, (Py_ssize_t)i))){
				spec->refine.include.keys[i] = PyString_AsString(PyList_GetItem(includeKeys, (Py_ssize_t)i));
			}else{
				PyErr_SetString(PyExc_TypeError, "Include keys must each be specified as a string.\nFound one that was not.\n");
				return NULL;
			}
		}
	}
	if (excludeKeys){
		if (!PyList_Check(excludeKeys)){
			PyErr_SetString(PyExc_TypeError, "Exclude keys must be specified as a list of strings.\n");
			return NULL;
		}
		spec->refine.flags = spec->refine.flags | ADB_REFINE_EXCLUDE_KEYLIST;
		spec->refine.exclude.nkeys = (uint32_t)PyList_Size(excludeKeys);
		spec->refine.exclude.keys = (const char **)calloc(sizeof(const char *), spec->refine.exclude.nkeys);
		for (i=0;i<spec->refine.exclude.nkeys;i++){
			 if (PyString_Check(PyList_GetItem(excludeKeys, (Py_ssize_t)i))){
				spec->refine.exclude.keys[i] = PyString_AsString(PyList_GetItem(excludeKeys, (Py_ssize_t)i));
			}else{
				PyErr_SetString(PyExc_TypeError, "Exclude keys must each be specified as a string.\nFound one that was not.\n");
				return NULL;
			}
		}
	}
	//the rest of spec->refine 
	if (radius){
		spec->refine.flags = spec->refine.flags | ADB_REFINE_RADIUS;
		spec->refine.radius = radius;
	}
	if (absThres){
		spec->refine.flags = spec->refine.flags | ADB_REFINE_ABSOLUTE_THRESHOLD;
		spec->refine.absolute_threshold = absThres;
	}
	if (relThres){
		spec->refine.flags = spec->refine.flags | ADB_REFINE_RELATIVE_THRESHOLD;
		spec->refine.relative_threshold = relThres;
	}
	if (durRatio){
		spec->refine.flags = spec->refine.flags | ADB_REFINE_DURATION_RATIO;
		spec->refine.duration_ratio = durRatio;
	}
	if (hop){
		spec->refine.flags = spec->refine.flags | ADB_REFINE_HOP_SIZE;
                /* not ideal but a temporary bandage fix */
		spec->refine.qhopsize = hop;
		spec->refine.ihopsize = hop;
	}
	//setup the datum
	spec->qid.datum->data = NULL;
	spec->qid.datum->power = NULL;
	spec->qid.datum->times = NULL;
	//grab the datum from the key
	ok = audiodb_retrieve_datum(current_db, key, spec->qid.datum);
	if (ok != 0){
		PyErr_SetString(PyExc_RuntimeError, "Encountered an error while trying to retrieve the data associated with the passed key.\n");
		return NULL;
	}
	result = audiodb_query_spec(current_db, spec);
	if (result == NULL){
		PyErr_SetString(PyExc_RuntimeError, "Encountered an error while running the actual query, or there was nothing returned.\n");
		return NULL;
		}
	if(strcmp(resFmt, "dict")==0){
		outgoing  = PyDict_New();
		for (i=0;i<result->nresults;i++){
			thisKey = PyString_FromString(result->results[i].ikey);
			if (!PyDict_Contains(outgoing, thisKey)){
				newBits =  Py_BuildValue("[(dII)]",
											result->results[i].dist, 
											result->results[i].qpos, 
											result->results[i].ipos);
				if (PyDict_SetItem(outgoing, thisKey,newBits)){
					printf("key : %s\ndist : %f\nqpos : %i\nipos : %i\n", result->results[i].ikey, result->results[i].dist, result->results[i].qpos, result->results[i].ipos);
					PyErr_SetString(PyExc_AttributeError, "Error adding a tuple to the result dict\n");
					Py_XDECREF(newBits);
					return NULL;
				}
				Py_DECREF(newBits);
			}else {
				//the key already has a value, so we need to fetch the value, confirm it's a list and append another tuple to it.
				currentValue = PyDict_GetItem(outgoing, thisKey);
				if (!PyList_Check(currentValue)){
					PyErr_SetString(PyExc_TypeError, "The result dictionary appears to be malformed.\n");
					return NULL;
				}
				newBits = Py_BuildValue("dII",result->results[i].dist, 
											result->results[i].qpos, 
											result->results[i].ipos);
				if (PyList_Append(currentValue,  newBits)){
					//error msg here
					Py_XDECREF(newBits);
					return NULL;
				}
				if (PyDict_SetItem(outgoing, thisKey, newBits)){
					PyErr_SetString(PyExc_AttributeError, "Error adding a tuple to the result dict\n");
					Py_XDECREF(newBits);
					return NULL;
				}
				Py_DECREF(newBits);
		
			}
		}
	}else if(strcmp(resFmt, "list")==0){
		outgoing  = PyList_New((Py_ssize_t)0);
		for (i=0;i<result->nresults;i++){
			newBits = Py_BuildValue("sdII",result->results[i].ikey,
										result->results[i].dist, 
										result->results[i].qpos, 
										result->results[i].ipos);
			if (PyList_Append(outgoing,  newBits)){
				//error msg here
				Py_XDECREF(newBits);
				return NULL;
			}
			Py_DECREF(newBits);
		}
		if(PyList_Reverse(outgoing)){//need to do this as things come off the accumulator backward.
			PyErr_SetString(PyExc_RuntimeError,
			"the reverse failed, hopefully a sensable error will follow.\nIf not, fix it.\n");
			return NULL;
			}
	}else{
		PyErr_SetString(PyExc_ValueError, 
			"Poorly specified result mode. Result must be either \'dist\' or \'list\'.\n");
		return NULL;
	}
	if (audiodb_query_free_results(current_db, spec, result)){
		printf("bit of trouble freeing the result and spec...\ncheck for leaks.");
	}
	
	return outgoing;
	
	
	
}

/* Data query.  
 * Returns a dict that is result ordered and key = result key
 * value is a list of tuples one per result associated with that key, of the form:
 *   (dist, qpos, ipos)
 * api call:
 * adb_query_results_t *audiodb_query_spec(adb_t *, const adb_query_spec_t *);
 ***/
PyObject * _pyadb_queryFromData(PyObject *self, PyObject *args, PyObject *keywds)
{
	adb_t *current_db;
	adb_query_spec_t *spec;
	adb_query_results_t *result;
	int ok, exhaustive, falsePositives;
	uint32_t i;
	const char *accuMode = "db";
	const char *distMode = "dot";
	const char *resFmt = "dict";
	uint32_t hop = 0;
	double radius = 0;
	double absThres = 0; 
	double relThres = 0; 
	double durRatio = 0;
	PyObject *includeKeys = NULL;
	PyObject *excludeKeys = NULL;
	PyObject *incoming = NULL;
	PyObject *outgoing = NULL;
	PyObject *thisKey = NULL;
	PyObject *currentValue = NULL;
	PyObject *newBits = NULL;
	unsigned int nDims = 0;
	unsigned int nVect = 0;
	PyArrayObject *features = NULL;
	PyArrayObject *power = NULL;
	PyArrayObject *times = NULL;
	adb_status_t *status;

	static char *kwlist[]  = { "db", "features", 
				   "seqLength", 
				   "seqStart", 
				   "exhaustive", 
				   "falsePositives",
				   "accumulation",
				   "distance",
				   "npoints",//nearest neighbor points per track
				   "ntracks",
				   "includeKeys",
				   "excludeKeys",
				   "radius",
				   "absThres",
				   "relThres",
				   "durRatio",
				   "hopSize",
				   "resFmt",
				   "power",
				   "times",
				   NULL
	};

	spec = (adb_query_spec_t *)malloc(sizeof(adb_query_spec_t));
	spec->qid.datum = (adb_datum_t *)malloc(sizeof(adb_datum_t));
	result = (adb_query_results_t *)malloc(sizeof(adb_query_results_t));
	
	spec->qid.sequence_length = 16;
	spec->qid.sequence_start = 0;
	spec->qid.flags = 0;
	spec->params.npoints = 1;
	spec->params.ntracks = 100;//number of results returned in db mode
	spec->refine.flags = 0;
	
	ok =  PyArg_ParseTupleAndKeywords(args, keywds, "OO!|iiiissIIOOddddIsO!O!", kwlist, 
					  &incoming, &PyArray_Type, &features, 
					  &spec->qid.sequence_length, 
					  &spec->qid.sequence_start, 
					  &exhaustive, &falsePositives,
					  &accuMode,&distMode,
					  &spec->params.npoints,
					  &spec->params.ntracks,
					  &includeKeys, &excludeKeys,
					  &radius, &absThres, &relThres, &durRatio, &hop,
					  &resFmt, 
					  &PyArray_Type, &power, &PyArray_Type, &times
												);
	
	if (!ok) {return NULL;}
	current_db = (adb_t *)PyCObject_AsVoidPtr(incoming);

	if (!features){ /* Sanity Check */
	  PyErr_SetString(PyExc_ValueError, 
			  "queryFromData: function requires feature data as numpy ndarray. PythonC required keyword check failed.\n");
	  return NULL;
	}

	/* Check the dimensionality of passed data agrees with the passed database */
	if(PyArray_NDIM(features)!=2){
	  PyErr_SetString(PyExc_ValueError, 
			  "queryFromData: passed features have incorrect shape, should be (nVecs, nDims).\n");
	  return NULL;
	}


	if(power && PyArray_NDIM(power)!=1){
	  PyErr_SetString(PyExc_ValueError, 
			  "queryFromData: passed power have incorrect shape, should be (nVecs,).\n");
	  return NULL;
	}

	if(times && PyArray_NDIM(times)!=1){
	  PyErr_SetString(PyExc_ValueError, 
			  "queryFromData: passed times have incorrect shape, should be (nVecs,).\n");
	  return NULL;
	}

	status = (adb_status_t*) malloc(sizeof(adb_status_t));
	int errtest = audiodb_status(current_db, status);
	if(errtest){
	  PyErr_SetString(PyExc_TypeError, "queryFromData failed: could not get status of passed ADB database");
	  free(status);
	  return NULL;
	}

	if(!PyArray_DIMS(features)[1]==status->dim){
	  PyErr_SetString(PyExc_ValueError, 
			  "queryFromData: passed features have incorrect dimensionality.\n");
	  free(status);
	  return NULL;
	}

	if(power && PyArray_DIMS(power)[0] != PyArray_DIMS(features)[0]){
	  PyErr_SetString(PyExc_ValueError, 
			  "queryFromData: passed power and features have incompatible nVecs dimension.\n");
	  free(status);
	  return NULL;
	}

	if(times && PyArray_DIMS(times)[0] != PyArray_DIMS(features)[0]){
	  PyErr_SetString(PyExc_ValueError, 
			  "queryFromData: passed times and features have incompatible nVecs dimension.\n");
	  free(status);
	  return NULL;
	}

	free(status);
	
	if (exhaustive){
		spec->qid.flags = spec->qid.flags | ADB_QID_FLAG_EXHAUSTIVE;
	}
	if (falsePositives){
		spec->qid.flags = spec->qid.flags | ADB_QID_FLAG_ALLOW_FALSE_POSITIVES;
	}
	
	//set up spec->params
	if (strcmp(accuMode,"db")==0){
		spec->params.accumulation = ADB_ACCUMULATION_DB;
	} else if (strcmp(accuMode,"track")==0){
		spec->params.accumulation = ADB_ACCUMULATION_PER_TRACK;
	} else if (strcmp(accuMode,"one2one")==0){
		spec->params.accumulation = ADB_ACCUMULATION_ONE_TO_ONE;
	} else{
		PyErr_SetString(PyExc_ValueError, 
			"Poorly specified distance mode. distance must either be \'db\', \'track\' or  \'one2one\'.\n");
		return NULL;
	}
	if (strcmp(distMode, "dot")==0){
		spec->params.distance = ADB_DISTANCE_DOT_PRODUCT;
	}else if (strcmp(distMode, "eucNorm")==0){
		spec->params.distance = ADB_DISTANCE_EUCLIDEAN_NORMED;
	}else if (strcmp(distMode, "euclidean")==0){
		spec->params.distance = ADB_DISTANCE_EUCLIDEAN;
	}else if (strcmp(distMode, "kullback")==0){
		spec->params.distance = ADB_DISTANCE_KULLBACK_LEIBLER_DIVERGENCE;
	}else{
		PyErr_SetString(PyExc_ValueError, 
			  "Poorly specified distance mode. distance must either be \'dot\', \'eucNorm\' ,\'euclidean\' or \'kullback\'.\n");
		return NULL;
	}
	
	//set up spec->refine
	//include/exclude keys
	if (includeKeys){
		if (!PyList_Check(includeKeys)){
			PyErr_SetString(PyExc_TypeError, "Include keys must be specified as a list of strings.\n");
			return NULL;
		}
		spec->refine.flags = spec->refine.flags | ADB_REFINE_INCLUDE_KEYLIST;
		spec->refine.include.nkeys = (uint32_t)PyList_Size(includeKeys);
		spec->refine.include.keys = (const char **)calloc(sizeof(const char *), spec->refine.include.nkeys);
		for (i=0;i<spec->refine.include.nkeys;i++){
			 if (PyString_Check(PyList_GetItem(includeKeys, (Py_ssize_t)i))){
				spec->refine.include.keys[i] = PyString_AsString(PyList_GetItem(includeKeys, (Py_ssize_t)i));
			}else{
				PyErr_SetString(PyExc_TypeError, "Include keys must each be specified as a string.\nFound one that was not.\n");
				return NULL;
			}
		}
	}
	if (excludeKeys){
		if (!PyList_Check(excludeKeys)){
			PyErr_SetString(PyExc_TypeError, "Exclude keys must be specified as a list of strings.\n");
			return NULL;
		}
		spec->refine.flags = spec->refine.flags | ADB_REFINE_EXCLUDE_KEYLIST;
		spec->refine.exclude.nkeys = (uint32_t)PyList_Size(excludeKeys);
		spec->refine.exclude.keys = (const char **)calloc(sizeof(const char *), spec->refine.exclude.nkeys);
		for (i=0;i<spec->refine.exclude.nkeys;i++){
			 if (PyString_Check(PyList_GetItem(excludeKeys, (Py_ssize_t)i))){
				spec->refine.exclude.keys[i] = PyString_AsString(PyList_GetItem(excludeKeys, (Py_ssize_t)i));
			}else{
				PyErr_SetString(PyExc_TypeError, "Exclude keys must each be specified as a string.\nFound one that was not.\n");
				return NULL;
			}
		}
	}
	//the rest of spec->refine 
	if (radius){
		spec->refine.flags = spec->refine.flags | ADB_REFINE_RADIUS;
		spec->refine.radius = radius;
	}
	if (absThres){
		spec->refine.flags = spec->refine.flags | ADB_REFINE_ABSOLUTE_THRESHOLD;
		spec->refine.absolute_threshold = absThres;
	}
	if (relThres){
		spec->refine.flags = spec->refine.flags | ADB_REFINE_RELATIVE_THRESHOLD;
		spec->refine.relative_threshold = relThres;
	}
	if (durRatio){
		spec->refine.flags = spec->refine.flags | ADB_REFINE_DURATION_RATIO;
		spec->refine.duration_ratio = durRatio;
	}
	if (hop){
		spec->refine.flags = spec->refine.flags | ADB_REFINE_HOP_SIZE;
                /* not ideal but a temporary bandage fix */
		spec->refine.qhopsize = hop;
		spec->refine.ihopsize = hop;
	}

	spec->qid.datum->data = (double*) features->data;
	
	if (power){
	  spec->qid.datum->power = (double*) power->data;
	}else{
	  spec->qid.datum->power=NULL;
	}
	
	if (times){
	  spec->qid.datum->times = (double*) times->data;
	}else{
	  spec->qid.datum->times=NULL;
	}

	nVect = PyArray_DIMS(features)[0];
	nDims = PyArray_DIMS(features)[1];
	spec->qid.datum->nvectors = (uint32_t)nVect;
	spec->qid.datum->dim = (uint32_t)nDims;

	result = audiodb_query_spec(current_db, spec);

	if (result == NULL){
		PyErr_SetString(PyExc_RuntimeError, "Encountered an error while running the actual query, or there was nothing returned.\n");
		return NULL;
		}
	if(strcmp(resFmt, "dict")==0){
		outgoing  = PyDict_New();
		for (i=0;i<result->nresults;i++){
			thisKey = PyString_FromString(result->results[i].ikey);
			if (!PyDict_Contains(outgoing, thisKey)){
				newBits =  Py_BuildValue("[(dII)]",
											result->results[i].dist, 
											result->results[i].qpos, 
											result->results[i].ipos);
				if (PyDict_SetItem(outgoing, thisKey,newBits)){
					printf("key : %s\ndist : %f\nqpos : %i\nipos : %i\n", result->results[i].ikey, result->results[i].dist, result->results[i].qpos, result->results[i].ipos);
					PyErr_SetString(PyExc_AttributeError, "Error adding a tuple to the result dict\n");
					Py_XDECREF(newBits);
					return NULL;
				}
				Py_DECREF(newBits);
			}else {
				//the key already has a value, so we need to fetch the value, confirm it's a list and append another tuple to it.
				currentValue = PyDict_GetItem(outgoing, thisKey);
				if (!PyList_Check(currentValue)){
					PyErr_SetString(PyExc_TypeError, "The result dictionary appears to be malformed.\n");
					return NULL;
				}
				newBits = Py_BuildValue("dII",result->results[i].dist, 
											result->results[i].qpos, 
											result->results[i].ipos);
				if (PyList_Append(currentValue,  newBits)){
					//error msg here
					Py_XDECREF(newBits);
					return NULL;
				}
				if (PyDict_SetItem(outgoing, thisKey, newBits)){
					PyErr_SetString(PyExc_AttributeError, "Error adding a tuple to the result dict\n");
					Py_XDECREF(newBits);
					return NULL;
				}
				Py_DECREF(newBits);
		
			}
		}
	}else if(strcmp(resFmt, "list")==0){
		outgoing  = PyList_New((Py_ssize_t)0);
		for (i=0;i<result->nresults;i++){
			newBits = Py_BuildValue("sdII",result->results[i].ikey,
										result->results[i].dist, 
										result->results[i].qpos, 
										result->results[i].ipos);
			if (PyList_Append(outgoing,  newBits)){
				//error msg here
				Py_XDECREF(newBits);
				return NULL;
			}
			Py_DECREF(newBits);
		}
		if(PyList_Reverse(outgoing)){//need to do this as things come off the accumulator backward.
			PyErr_SetString(PyExc_RuntimeError,
			"the reverse failed, hopefully a sensable error will follow.\nIf not, fix it.\n");
			return NULL;
			}
	}else{
		PyErr_SetString(PyExc_ValueError, 
			"Poorly specified result mode. Result must be either \'dist\' or \'list\'.\n");
		return NULL;
	}
	if (audiodb_query_free_results(current_db, spec, result)){
		printf("bit of trouble freeing the result and spec...\ncheck for leaks.");
	}
	
	return outgoing;
	
	
	
}


/* retrieval of inserted data 
* returned numpy array has ndarray.shape = (numVectors, numDims)
* array datatype needs to be doubles (float may work...)
* if power reqeusted, it will be a 1d array of length numVectors
* if times are requested, they will be a 1d array of length 2*nvectors
*/

// api call: 
// typedef struct adb_datum {
//   uint32_t nvectors;
//   uint32_t dim;
//   const char *key;
//   double *data;
//   double *power;
//   double *times;
// } adb_datum_t;

//int audiodb_retrieve_datum(adb_t *, const char *, adb_datum_t *);
//int audiodb_free_datum(adb_t *, adb_datum_t *);
PyObject * _pyadb_retrieveDatum(PyObject *self, PyObject *args, PyObject *keywds)
{
	adb_t *current_db = NULL;
	adb_status_t *status = NULL;
	adb_datum_t *ins = NULL;
	int ok=0, errtest=0;
	unsigned features=0, powers=0, times=0;
	PyObject *incoming = 0; // The ADB database
	PyObject *outgoing = 0; // The PyArrayObject
	const char *key = NULL;
	static char *kwlist[]  = { "db", "key", "features", "powers", "times", NULL};
	double * data = NULL;
	int dims = 0;
	npy_intp shape[2] = { 0, 0 };	

	ok =  PyArg_ParseTupleAndKeywords(args, keywds, "Os|III", kwlist, &incoming, &key, &features, &powers, &times);
	if (!ok){
	  PyErr_SetString(PyExc_TypeError, "Failed at PyArg_ParseTupleAndKeywords");
	  return NULL;
	}

	if(features+powers+times>1){
	  PyErr_SetString(PyExc_TypeError, "Failed: you must specify only one of features, powers, or times");
	  return NULL;	  
	}

	if(!(features||powers||times)){
	  features=1; // default is to return features
	}

	current_db = (adb_t *)PyCObject_AsVoidPtr(incoming);
	if (!current_db){
	  PyErr_SetString(PyExc_TypeError, "Failed to convert open database to C-pointer");
	  return NULL;
	}
	status = (adb_status_t*) malloc(sizeof(adb_status_t));
	errtest = audiodb_status(current_db, status);
	if(errtest){
	  PyErr_SetString(PyExc_TypeError, "Failed: could not get status of passed ADB database");
	  free(status);
	  return NULL;
	}

	if(powers && !(status->flags&ADB_HEADER_FLAG_POWER)){
	  PyErr_SetString(PyExc_TypeError, "Failed: powers requested but passed ADB database has no powers");
	  free(status);
	  return NULL;
	}

	if(times && !(status->flags&ADB_HEADER_FLAG_TIMES)){
	  PyErr_SetString(PyExc_TypeError, "Failed: times requested but passed ADB database has no times");
	  free(status);
	  return NULL;
	}	

	ins = (adb_datum_t *)malloc(sizeof(adb_datum_t));
	errtest = audiodb_retrieve_datum(current_db, key, ins); // retrieve data from adb via key
	if (errtest){
	  PyErr_SetString(PyExc_TypeError, "Failed to retrieve datum");
	  free(ins);
	  return NULL;
	}

	if(features){
	  if(ins->dim>1){
	    dims=2;
	    shape[1]= ins->dim;
	  }
	  else{
	    dims=1;
	  }
	  shape[0]= ins->nvectors;
	  data = ins->data;
	}
	else if(powers){
	  dims=1;
	  shape[0]= ins->nvectors;
	  data = ins->power;
	}
	else if(times){
	  dims=1;
	  shape[0]= 2 * ins->nvectors;
	  data = ins->times;
	}

	outgoing = PyArray_SimpleNew(dims, shape, NPY_DOUBLE);
	if (!outgoing){
	  free(status);
	  free(ins); // free the malloced adb_datum_t structure though
	  Py_XDECREF(outgoing);
	  PyErr_SetString(PyExc_TypeError, "Failed to convert retrieved datum to C-Array");
	  return NULL;
	}	

	/* Copy the data, this allows us to free the allocated memory and let
	 * python do the subsequent garbage collection itself.
	 */
	int num_items = ins->nvectors;
	if(dims>1){
	  num_items *= shape[1];
	}		    
	double* p = (double*) PyArray_DATA(outgoing);
	double* d = data;
	while(num_items--)
	  *p++ = *d++;	
	audiodb_free_datum(current_db, ins); // free the source audiodb_datum
	free(status); // free the malloced status object
	free(ins); // free the malloced adb_datum_t structure though
	return outgoing; 
}


/* close a database */
/* api call: */
// void audiodb_close(adb_t *db);
static void _pyadb_close(void *ptr)
{
	adb_t *stale_database;
	stale_database = (adb_t *)ptr; 
	
	audiodb_close(stale_database);
}

static PyMethodDef _pyadbMethods[] = 
{
	{ "_pyadb_create", _pyadb_create, METH_VARARGS, 
	  "_pyadb_create(string path, unsigned datasize, unsigned ntracks, unsigned datadim)->adb object"},
	{ "_pyadb_open", _pyadb_open, METH_VARARGS, 
	  "_pyadb_open(string path, [\'r\'|\'w\'])->adb object\nNote that specifing \'w\' opens the file in read and write mode.  \
	There is currently no way to open in write only."},
	{ "_pyadb_status", _pyadb_status, METH_VARARGS,
	  "_status(adb_t *)->(numFiles, dims, dudCount, nullCount, flags, length, data_region_size)"},
	{ "_pyadb_l2norm", _pyadb_l2norm, METH_VARARGS,
	  "_pyadb_l2norm(adb_t *)->int return code (0 for sucess)"},
	{ "_pyadb_power", _pyadb_power, METH_VARARGS,
	  "_pyadb_power(adb_t *)->int return code (0 for sucess)"},
	{"_pyadb_insertFromArray", (PyCFunction)_pyadb_insertFromArray, METH_VARARGS | METH_KEYWORDS,
	"_pyadb_insertFromArray(adb_t *, features=ndarray, [power=ndarray | key=keystring | times=ndarray])->\
	int return code (0 for sucess)\n\
	insert feature data from a numpy array\n\
	array given should have ndarray.shape = (numDims*numVectors,)\n\
	array datatype needs to be doubles (float may work...)\n\
	if power is given, must be 1d array of length numVectors\n\
	if times is given, must be 1d array of length 2*numVectors like this:\n\
	int audiodb_insert_datum(adb_t *, const adb_datum_t *);"},
	{"_pyadb_retrieveDatum", (PyCFunction)_pyadb_retrieveDatum, METH_VARARGS | METH_KEYWORDS, "_pyadb_retrieveDatum(adb_t *, key=keystring"},
	{ "_pyadb_insertFromFile", (PyCFunction)_pyadb_insertFromFile, METH_VARARGS | METH_KEYWORDS,
	  "_pyadb_insertFromFile(adb_t *, features=featureFile, [power=powerfile | key=keystring | times=timingFile])->\
	int return code (0 for sucess)"},
	{ "_pyadb_liszt", (PyCFunction)_pyadb_liszt, METH_VARARGS,
	  "_pyadb_liszt(adb_t*)->[[key1,numvecs1],[key2,numvecs2]...]"},
	{ "_pyadb_queryFromKey", (PyCFunction)_pyadb_queryFromKey, METH_VARARGS | METH_KEYWORDS,
	 "base query.  The nomenclature here is about a far away as pythonic as is possible.\n\
This should be taken care of via the higher level python structure\n\
returns a dict that should be result ordered and key = result key\n\
and value is a list of tuples one per result associated with that key, of the form:\n\
   \t(dist, qpos, ipos)\n\
Note as well that this is by no means the most efficient way to cast from C, simply the most direct\n\
and what it lacks in effeciency it gains in python side access.  It remains to be seen if this is\n\
a sensible trade.\n\
_pyadb_queryFromKey(adb_t *, query key,\n\
					[seqLength    = Int Sequence Length, \n\
					seqStart      = Int offset from start for key, \n\
					exhaustive    = boolean - True for exhaustive (false by default),\n\
					falsePositives= boolean - True to keep fps (false by defaults),\n\
					accumulation  = [\"db\"|\"track\"|\"one2one\"] (\"db\" by default),\n\
					distance      = [\"dot\"|\"eucNorm\"|\"euclidean\"] (\"dot\" by default),\n\
					npoints       = int number of points per track,\n\
					ntracks       = max number of results returned in db accu mode,\n\
					includeKeys   = list of strings to include (use all by default),\n\
					excludeKeys   = list of strings to exclude (none by default),\n\
					radius        = double of nnRadius (1.0 default, overrides npoints if specified),\n\
					absThres      = double absolute power threshold (db must have power),\n\
					relThres      = double relative power threshold (db must have power),\n\
					durRatio      = double time expansion/compresion ratio,\n\
					hopSize       = int hopsize (1 by default)])->resultDict\n\
					resFmt        = [\"list\"|\"dict\"](\"dict\" by default)"},
	{"_pyadb_queryFromData", (PyCFunction)_pyadb_queryFromData, METH_VARARGS | METH_KEYWORDS,
	 "data query. Required features=F (numpy ndarray). Optional: power=P (numpy 1d array), times=T (numpy 1d array)"},
	{NULL,NULL, 0, NULL}
};

void init_pyadb(void)
{
	Py_InitModule3("_pyadb", _pyadbMethods, "internal c bindings for audioDB.  Use pyadb for pythonic access to adb.");
	import_array();
	return;
}

	
