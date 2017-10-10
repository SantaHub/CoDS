#!/usr/bin/python

####################MYSQL######################
import MySQLdb
db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="john",         # your username
                     passwd="megajonhy",  # your password
                     db="jonhydb")        # name of the data base
cur = db.cursor()
cur.execute("SELECT * FROM YOUR_TABLE_NAME")
for row in cur.fetchall():
    print row[0]
db.close()

###################
