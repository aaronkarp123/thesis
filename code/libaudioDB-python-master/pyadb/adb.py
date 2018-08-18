#!/usr/bin/env python
# encoding: utf-8
"""
pyadb.py

public access and class structure for python audioDb api bindings.



Created by Benjamin Fields on 2009-09-22.
Copyright (c) 2009 Goldsmith University of London.
"""

import sys
import os, os.path
import unittest
import _pyadb

ADB_HEADER_FLAG_L2NORM = 0x1      #annoyingly I can't find a means
ADB_HEADER_FLAG_POWER = 0x4       #around defining these flag definitions 
ADB_HEADER_FLAG_TIMES = 0x20      #as they aren't even exported to the 
ADB_HEADER_FLAG_REFERENCES = 0x40 #api, so this is the only way to get them.

class Usage(Exception):
	"""error to indicate that a method has been called with incorrect args"""
	def __init__(self, msg):
		self.msg = msg

class Pyadb(object):
	"""Pyadb class.  Allows for creation, access, insertion and query of an audioDB vector matching database."""
	validConfigTerms = {"seqLength":int, "seqStart":int, "exhaustive":bool, 
		"falsePositives":bool, "accumulation":str, "distance":str, "npoints":int,
		"ntracks":int, "includeKeys":list, "excludeKeys":list, "radius":float, "absThres":float,
		"relThres":float, "durRatio":float, "hopSize":int, "resFmt":str}
	def __init__(self, path, mode='w', datasize=0, ntracks=0, datadim=0):
		"""
		initialize the database.  By default db will hold 20000 tracks, be 2GB in size and determine datadim from the first inserted feature
		"""
		self.path = path
		self.configQuery = {}
		if not (mode=='w' or mode =='r'):
			raise(ValueError, "if specified, mode must be either\'r\' or \'w\'.")
		if os.path.exists(path):
			self._db = _pyadb._pyadb_open(path, mode)
		else:
			self._db = _pyadb._pyadb_create(path,datasize,ntracks,datadim)
		self._updateDBAttributes()
		return
	
	def insert(self, featFile=None, powerFile=None, timesFile=None, featData=None, powerData=None, timesData=None, key=None):
		"""
		Insert features into database.  Can be done with data provided directly or by giving a path to a binary fftExtract style feature file.  If power and/or timing is engaged in the database header, it must be provided (via the same means as the feature) or a Usage exception will be raised. Power files should be of the same binary type as features.  Times files should be the ascii number length of time in seconds from the begining of the file to segment start, one per line.
		If providing data directly, featData should be a numpy array with shape= (number of Dimensions, number of Vectors)
		"""
		#While python style normally advocates leaping before looking, these check are nessecary as 
		#it is very difficult to assertain why the insertion failed once it has been called.
		if (self.hasPower and (((featFile) and powerFile==None) or ((not featData==None) and powerData==None))):
			raise(Usage, "The db you are attempting an insert on (%s) expects power and you either\
 haven't provided any or have done so in the wrong format."%self.path)
		if (self.hasTimes and (((timesFile) and timesFile==None) or ((not timesData==None) and timesData==None))):
			raise(Usage, "The db you are attempting an insert on (%s) expects times and you either\
 haven't provided any or have done so in the wrong format."%self.path)
		args = {"db":self._db}
		if featFile:
			args["features"] = featFile
		elif (featData != None):
			args["features"] = featData
		else:
			raise(Usage, "Must provide some feature data!")
		if self.hasPower:
			if featFile:
				args["power"]=powerFile
			elif featData.any():
				args["power"]=powerData
		if timesData != None:
			self.hasTimes=True
		if self.hasTimes:
			if featFile:
				args["times"]=timesFile
			elif timesData.any():
				args["times"]=timesData
		if key:
			args["key"]=str(key)
		if featFile:
			if not _pyadb._pyadb_insertFromFile(**args):
				raise RuntimeError("Insertion from file failed for an unknown reason.")
			else:
				self._updateDBAttributes()
				return
		elif (featData != None):
			if (len(args["features"].shape) == 1) : 
				args["features"] = args["features"].reshape((args["features"].shape[0],1))
			args["nVect"], args["nDim"] = args["features"].shape
			args["features"] = args["features"].flatten()
			if(self.hasPower and powerData != None):
				if (len(args["power"].shape) == 1) : 
					args["power"] = args["power"].reshape((args["power"].shape[0],1))
				args["power"] = args["power"].flatten()
			if(self.hasTimes and timesData != None):
				if (len(args["times"].shape) == 1) : 
					args["times"] = args["times"].reshape((args["times"].shape[0],1))
				args["times"] = args["times"].flatten()

			ok = _pyadb._pyadb_insertFromArray(**args)
			if not (ok==0):
				raise RuntimeError("Direct data insertion failed for an unknown reason. err code = %i"%ok)
			else:
				self._updateDBAttributes()
				return
	
	def configCheck(self, scrub=False):
		"""examine self.configQuery dict.  For each key encouters confirm it is in the validConfigTerms list and if appropriate, type check.  If scrub is False, leave unexpected keys and values alone and return False, if scrub  try to correct errors (attempt type casts and remove unexpected entries) and continue.  If self.configQuery only contains expected keys with correctly typed values, return True.  See Pyadb.validConfigTerms for allowed keys and types.  Note also that include/exclude key lists memebers or string switched are not verified here, but rather when they are converted to const char * in the C api call and if malformed, an error will be rasied from there.  Valid keys and values in  queryconfig:
		{seqLength    : Int Sequence Length, \n\
		seqStart      : Int offset from start for key, \n\
		exhaustive    : boolean - True for exhaustive (false by default),\n\
		falsePositives: boolean - True to keep fps (false by defaults),\n\
		accumulation  : [\"db\"|\"track\"|\"one2one\"] (\"db\" by default),\n\
		distance      : [\"dot\"|\"eucNorm\"|\"euclidean\"|\"kullback\"] (\"dot\" by default),\n\
		npoints       : int number of points per track,\n\
		ntracks       : max number of results returned in db accu mode,\n\
		includeKeys   : list of strings to include (use all by default),\n\
		excludeKeys   : list of strings to exclude (none by default),\n\
		radius        : double of nnRadius (1.0 default, overrides npoints if specified),\n\
		absThres      : double absolute power threshold (db must have power),\n\
		relThres      : double relative power threshold (db must have power),\n\
		durRatio      : double time expansion/compresion ratio,\n\
		hopSize       : int hopsize (1 by default)])->resultDict\n\
		resFmt        : [\"list\"|\"dict\"](\"dict\" by default)}"""
		for key in self.configQuery.keys():
			if key not in Pyadb.validConfigTerms.keys():
				if not scrub: return False
				print "scrubbing %s from query config."%str(key)
				del self.configQuery[key]
			if not isinstance(self.configQuery[key], Pyadb.validConfigTerms[key]):
				if not scrub: return False
				self.configQuery[key] = Pyadb.validConfigTerms[key](self.configQuery[key])#hrm, syntax?
		return True	
				
				# 
	
	def query(self, key=None, featData=None, strictConfig=True):
		"""query the database.  Query parameters as defined in self.configQuery. For details on this consult the doc string in the configCheck method."""
		if not self.configCheck():
			if strictConfig:
				raise ValueError("configQuery dict contains unsupported terms and strict configure mode is on.\n\
Only keys found in Pyadb.validConfigTerms may be defined")
			else:
				print "configQuery dict contains unsupported terms and strict configure mode is off.\n\
Only keys found in Pyadb.validConfigTerms should be defined.  Removing invalid terms and proceeding..."
				self.configCheck(scrub=True)
		if ((not key and not featData) or (key and featData)):
			raise Usage("query require either key or featData to be defined, you have defined both or neither.")
		if key:
			result = _pyadb._pyadb_queryFromKey(self._db, key, **self.configQuery)
		elif featData:
			raise NotImplementedError("direct data query not yet implemented.  Sorry.")
		return Pyadb.Result(result, self.configQuery)

	def query_data(self, featData=None, powerData=None, timesData=None, strictConfig=True):
		"""query the database using numpy arrays. required data: featData, optional data: [powerData, timesData]Query parameters as defined in self.configQuery. For details on this consult the doc string in the configCheck method."""
		if not self.configCheck():
			if strictConfig:
				raise ValueError("configQuery dict contains unsupported terms and strict configure mode is on.\n\
Only keys found in Pyadb.validConfigTerms may be defined")
			else:
				print "configQuery dict contains unsupported terms and strict configure mode is off.\n\
Only keys found in Pyadb.validConfigTerms should be defined.  Removing invalid terms and proceeding..."
				self.configCheck(scrub=True)
		cq = self.configQuery.copy()
		if (featData==None):
			raise Usage("query requires featData to be defined.")
		if(powerData!=None):
			cq['power']=powerData
		if(timesData!=None):
			cq['times']=timesData
		result = _pyadb._pyadb_queryFromData(self._db, featData, **cq)
		return Pyadb.Result(result, self.configQuery)
	
	def status(self):
		'''update attributes and return them as a dict'''
		self._updateDBAttributes()
		return {	"numFiles" : self.numFiles, 
					"dims"     : self.dims, 
					"dudCount" : self.dudCount, 
					"nullCount": self.nullCount, 
					"length"   : self.length, 
					"data_region_size" : self.data_region_size,
					"l2Normed" : self.l2Normed,
					"hasPower" : self.hasPower,
					"hasTimes" : self.hasTimes, 
					"usesRefs" : self.usesRefs}
	###internal methods###
	def _updateDBAttributes(self):
		'''run _pyadb_status to fill/update the database level flags and info'''
		rawFlags = long(0)
		(self.numFiles, 
		self.dims, 
		self.dudCount, 
		self.nullCount, 
		rawFlags, 
		self.length, 
		self.data_region_size) = _pyadb._pyadb_status(self._db)
		self.l2Normed = bool(rawFlags & ADB_HEADER_FLAG_L2NORM)
		self.hasPower = bool(rawFlags & ADB_HEADER_FLAG_POWER)
		self.hasTimes = bool(rawFlags & ADB_HEADER_FLAG_TIMES)
		self.usesRefs = bool(rawFlags & ADB_HEADER_FLAG_REFERENCES)
		return
		
	class Result(object):
		def __init__(self, rawData, currentConfig):
			self.rawData = rawData
			if "resFmt" in currentConfig:
				self.type = currentConfig["resFmt"]
			else:
				self.type = "dict"
		def __str__(self):
			return str(self.rawData)
		def __repr__(self):
			return repr(self.rawData)

	def liszt(self):
		'''run _pyadb_liszt to get a list of database keys'''
		if self._db != None:
			return _pyadb._pyadb_liszt(self._db)
		else:
			print "Error in liszt(): ADB database not defined"
			return 0

	def retrieve_datum(self, key, **args):
		'''run _pyadb_retrieveDatum to retrieve data by key:
		      features=True, to get features
		      powers=True, to get Powers
		      times=True, to get Times
		'''
		if self._db != None:
			return _pyadb._pyadb_retrieveDatum(self._db, key=key, **args)
		else:
			print "Error in liszt(): ADB database not defined"
			return 0
		
class untitledTests(unittest.TestCase):
	def setUp(self):
		pass


if __name__ == '__main__':
	unittest.main()
