import os

from pymongo import MongoClient

conn = os.environ['MONGODB_CONNSTRING']

CLIENT = MongoClient(conn)
DATABASE = CLIENT["ALLiS"]
COLL_PREDICTION = DATABASE['PREDICTION']
COLL_CFPOOL = DATABASE['CFPOOL']
COLL_CAMPAIGN = DATABASE['CAMPAIGN']
COLL_REACTION = DATABASE['REACTION']
COLL_LIGAND = DATABASE['LIGAND']
COLL_MODEL = DATABASE['MODEL']

MongoHasData = True
for init_coll in (COLL_LIGAND, COLL_REACTION, COLL_MODEL):
    if init_coll.find_one() is None:
        MongoHasData = False
        break


