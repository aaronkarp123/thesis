#!/usr/bin/python

from pyadb.adb import *
import web
import json
import sys
import getopt

# DB Path goes here for now!
dbPath = "9.adb"

urls = (
	'/', 'index',
	'/status', 'status',
	'/query', 'query'
)

app = web.application(urls, globals())
class index:
	def GET(self):
		return """
<html>
<head><title>audioDB server</title></head>
<body>
<ul>
<h2>Path: """+dbPath+"""</h2>
<li><a href="/status">Status</a></li>
<li><a href="/query">Query</a></li>
</ul>
</body>
</html>"""


class status:
	def GET(self):
		web.header("Content-Type", "application/json") 

		db = Pyadb(path = dbPath, mode = "r")
		results = db.status()
		return json.dumps(dict(status = "ok", data = results))

class query:
	def GET(self):
		web.header("Content-Type", "application/json") 
		params = web.input(key="", ntracks=100, seqStart=0, seqLength=16, npoints=1, radius=1.0, hopSize=1, exhaustive=False, falsePositives=False, accumulation="db", distance="dot", absThres=0, relThres=0, durRatio=0, includeKeys=[], excludeKeys=[])
		results = dict()
		db = Pyadb(path = dbPath, mode = "r")
	
		if not params.includeKeys == []:
			db.configQuery["includeKeys"] = map(str, params.includeKeys)
		
		if params.excludeKeys:
			foo = map(str, params.excludeKeys)
			db.configQuery["excludeKeys"] = foo 

		db.configQuery["ntracks"] = int(params.ntracks)
		db.configQuery["npoints"] = int(params.npoints)
		db.configQuery["seqStart"] = int(params.seqStart)
		db.configQuery["seqLength"] = int(params.seqLength)
		db.configQuery["hopSize"] = int(params.hopSize)
		db.configQuery["radius"] = float(params.radius)
		db.configQuery["absThres"] = float(params.absThres)
		db.configQuery["relThres"] = float(params.relThres)
		db.configQuery["durRatio"] = float(params.durRatio)
		db.configQuery["resFmt"] = "list" 
		

		
		try:
			results = db.query(key = params.key)
		except Exception as inst:
			return json.dumps(dict(status = "error", message=str(inst)))
		return json.dumps(dict(status = "ok", data = results.rawData))

if __name__ == "__main__": 
	app.run()
