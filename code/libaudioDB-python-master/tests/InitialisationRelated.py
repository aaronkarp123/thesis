#!/usr/bin/env python
# encoding: utf-8
"""
InitialisationRelated.py

designed to mirror the numbering for the C/C++ api's unit tests
this performs tests 0001, 0002, 0003


Created by Ben Fields on 2010-01-11.
"""

import sys
import os,os.path
from pyadb import adb
import numpy as np
import struct
import unittest


class CreateADBTests(unittest.TestCase):
	def setUp(self):
		self.adb = adb.Pyadb("test.adb")
	def test_DBcreation(self):
		self.assert_(os.path.exists(self.adb.path))
		self.assertRaises(TypeError, adb.Pyadb)
	def test_DBstatus(self):
		try:
			self.adb.status()
		except:
			self.assert_(False)
	def test_1DinsertionFromFileSelfQuery(self):
		tH = open("testfeature", 'w')
		tH.write(struct.pack("=id",1,1))
		tH.close()
		self.adb.insert("testfeature", key='testfeature')
		self.adb.configQuery["seqLength"] = 1
		result = self.adb.query("testfeature")
		self.assert_(len(result.rawData) == 1)
		self.assert_(result.rawData.has_key("testfeature"))
		self.assert_(len(result.rawData["testfeature"]) == 1)
		self.assert_(result.rawData["testfeature"][0] == (1.0, 0,0))
		os.remove(self.adb.path)#delete the db
	def test_1DinsertionFromArraySelfQuery(self):
		test1 = np.ones(1)
		print "test1: " + str(test1)
		self.adb.insert(featData=test1, key="testfeature")
		self.adb.configQuery["seqLength"] = 1
		result = self.adb.query(key="testfeature")
		self.assert_(len(result.rawData) == 1)
		self.assert_(result.rawData.has_key("testfeature"))
		self.assert_(len(result.rawData["testfeature"]) == 1)
		self.assert_(result.rawData["testfeature"][0] == (1.0, 0,0))
		


if __name__ == '__main__':
	unittest.main()
