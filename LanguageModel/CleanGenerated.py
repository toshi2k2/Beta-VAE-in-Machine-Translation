import sys

EOS = "<eos>"

def WriteFile(line):
    name = "clean_monolingual.af"
    fp = open(name, 'a')
    fp.write(line)
    fp.close()


def clean(trunk_file):
    fp = open(trunk_file)
    line = []
    for word in fp.read().split():
        if word == EOS:
            if len(line) > 0:
                line.append('\n')
                sent = ' '.join(line)
                WriteFile(sent)
                line = []
        else:
            line.append(word) 

    fp.close



def main():
    arg = sys.argv[1]
    clean(arg)
if __name__ ==  "__main__":
    main()