#!/usr/bin/python
import os
from os.path import split,join
import sqlite3

with sqlite3.connect("samples.db") as con:
    cur = con.cursor()

    cur.execute('SELECT MAX("Index") FROM samples;')
    temp = cur.fetchone()[0]
    print(temp)
    if temp is None:
        index = 0
    else:
        index = int(temp) + 1

    for root,dirs,files in os.walk("/home/me1alw/Dropbox/Science/Birds/Jay"):
        if root.find("_") >= 0:
            continue
        images = [f for f in files if f[-4:] == ".jpg"]
        if len(images) == 0:
            continue
        index += 1
        print(split(root)[1])
        print(index)
        cur.execute('insert into samples ("index","name") values (?,?);',(index,split(root)[1]))
        #Find max key
        cur.execute('SELECT MAX("key") from images;')
        key = cur.fetchone()[0]
        if key is None:
            key = 0
        for i in images:
            key += 1
            cur.execute('insert into images ("key","image","bar","pixels","sample") values (?,?,?,?,?);',
                        (key,join(root,i),1,1,index))
            print(i)
