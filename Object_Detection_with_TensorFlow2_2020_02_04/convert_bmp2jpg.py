import os
#import subprocess
#import shlex
from os import sys
import subprocess


def ConvertToJPG(pathToFile, newpathToFile):
    p = subprocess.Popen(['convert', '-quality', '100', pathToFile, newpathToFile], stdout = subprocess.PIPE, stderr=subprocess.PIPE)

def getFiles(path):
    '''Gets all of the files in a directory'''
    sub = os.listdir(path)
    paths = {} # dictionary for stored filename, path.
    for p in sub:
        print("Where...", p) # just to see where the function has reached; it's optional
        pDir = os.path.join(path, p)
        if os.path.isdir(pDir):
            paths.update(getAllFiles(pDir, paths))
        else:
            paths[p] = pDir
    #print(paths)
    return paths

def man():

    try:
        imgPath = sys.argv[1]
    except NameError:
        #imgPath = 'xxxxxxx'
        print('sys.argv[1], the path of directory not giving!')
        exit()
        
    paths = getFiles(imgPath)
    for filename in paths:
        if filename.endswith(".bmp"):
            print("Check img name", imgPath, filename)
#            ConvertToJPG(pathToFile, newpathToFile)



# Run the kernal #
if __name__ == '__main__':
    man(沒改完有空再改囉)
    
    
