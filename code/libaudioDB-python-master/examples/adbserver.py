#!/usr/bin/python

import sqlite3
import csv
import sys
import urllib
import web
from pyadb.adb import *
import simplejson as json
import tempfile
import subprocess

# For now

featureFile = "AWAL2010.adb"
trackFile = "awal_id32.csv"
audioDir = "/home/mjewell/Music/mp3s/"

urls = (
	'/track/(.*)', 'trackSearch',
	'/audio/(.*)', 'audioPlay',
	'/search/(.*)', 'segmentSearch',
	'/crossdomain.xml', 'crossDomain'
)

app = web.application(urls, globals())
db = Pyadb(path = featureFile, mode = "r")
dbfile = tempfile.NamedTemporaryFile(suffix = ".db")
mdb = web.database(dbn='sqlite', db="metadata.db")



def buildDatabase(csvfile):
	print "Build DB"
	mdb._db_cursor().connection.text_factory=str
	mdb._db_cursor().connection.execute('DROP TABLE IF EXISTS media')
	mdb._db_cursor().connection.execute('CREATE TABLE media (uid TEXT NOT NULL PRIMARY KEY, artist TEXT, album TEXT, track TEXT, tracknum INTEGER, filename TEXT, seconds INTEGER)')
	
	trackReader = csv.reader(open(csvfile))
	head = True
	for row in trackReader:
		if head:
			head = False
			continue
		mdb.insert('media', uid=row[0], filename=row[1], artist=row[2], track=row[3], album=row[4], tracknum=row[5], seconds=row[6])

	
def retrieveTrackData(trackID):
	results = mdb.select('media', dict(uid=trackID), where='uid = $uid')
	res = dict()
	try:
		result = results[0]
	except IndexError:
		return False
	for key in result.keys():
		res[key] = result[key]

	return res

class trackSearch:
	def GET(self, trackID):
		data = retrieveTrackData(trackID)
		if data:
				return json.dumps(dict(status = "ok", data = data)) 
		else:
				return json.dumps(dict(status = "error", message=str("Invalid key")))

class crossDomain:
	def GET(self):
		return """<?xml version="1.0"?> 
<!DOCTYPE cross-domain-policy SYSTEM "http://www.macromedia.com/xml/dtds/cross-domain-policy.dtd"> 
<cross-domain-policy>	
	<allow-access-from domain="*" /> 
</cross-domain-policy> """

class audioPlay:
	def GET(self, trackID):
		web.header("Content-Type", "audio/mpeg")
		user_data = web.input()
		track_data = retrieveTrackData(trackID)
		tempMp3 = tempfile.NamedTemporaryFile(suffix = ".mp3")
		tempWav = tempfile.NamedTemporaryFile(suffix = ".wav")
		subprocess.call(["sox", audioDir+track_data['filename'], tempWav.name, "trim", user_data['start'], user_data['length'], "fade", "0.3", user_data['length']])
		subprocess.call(["lame", "--quiet", tempWav.name, tempMp3.name])

		return tempMp3.read() 

class segmentSearch:
	def GET(self, trackID):
		params = web.input(db="AWAL",key="", ntracks=20, seqStart=0, seqLength=16, npoints=1, radius=1.0, hopSize=1, exhaustive=False, falsePositives=False, accumulation="track", distance="eucNorm", absThres=-6, relThres=10, durRatio=0, includeKeys=[], excludeKeys=[], resFmt="dict")
		
		db.configQuery["ntracks"] = int(params.ntracks)
		db.configQuery["npoints"] = int(params.npoints)
		db.configQuery["seqStart"] = int(params.seqStart)
		db.configQuery["seqLength"] = int(params.seqLength)
		db.configQuery["hopSize"] = int(params.hopSize)
		db.configQuery["radius"] = float(params.radius)
		db.configQuery["absThres"] = float(params.absThres)
		db.configQuery["relThres"] = float(params.relThres)
		db.configQuery["durRatio"] = float(params.durRatio)
		db.configQuery["resFmt"] = "list";
		#db.configQuery["includeKeys"] = ["AWAL2000", "AWAL500", "AWAL1000"]

		results = dict()
		try:
			results = db.query(key = trackID) 
			return json.dumps(dict(status = "ok", data = results.rawData))
		except Exception:
			return json.dumps(dict(status = "error", message=str("Fix inst")))

if __name__ == "__main__":
	# Uncomment to rebuild the db at start
	#buildDatabase(trackFile)
	app.run()
