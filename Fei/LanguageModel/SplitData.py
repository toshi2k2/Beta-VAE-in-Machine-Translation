import sys
import random
import time

random.seed(time.time())

TestRate = 0.1
TrainRate = 0.8
# ValidRate = 0.1
def WriteFile(line,mode):
    if mode == "TEST":
        name = "test.txt"
    elif mode == "TRAIN":
        name = "train.txt"
    elif mode == "VALID":
        name = "valid.txt" 
    fp = open(name, 'a')
    fp.write(line)
    fp.close()


def split(trunk_file):
    
    with open(trunk_file,'r') as fp:
        line = fp.readline()
        line_cnt = 1
        
        while line :
            line = fp.readline()
            
            r = random.random()
            if r < TestRate:
                mode = "TEST"
            elif r < TestRate + TrainRate:
                mode = "TRAIN"
            else:
                mode = "VALID"
            
            WriteFile(line,mode)
            
            line_cnt += 1
            if(line_cnt % 10000 ==0):
                print("%d lines assigned."%line_cnt)

            line = fp.readline()
    fp.close


def main():
    org_file = sys.argv[1]
    split(org_file)
if __name__ ==  "__main__":
    main()