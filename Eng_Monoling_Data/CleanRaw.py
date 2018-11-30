import sys

EOS = {".","!","?"}


def WriteFile(line):
    name = "clean_monolingual.en"
    fp = open(name, 'a')
    fp.write(line)
    fp.close()

def main():
    raw_file = sys.argv[1]
    # print("Cleaning",raw_file)

    fp = open(raw_file)
    line = []
    for word in fp.read().strip().split():
        line.append(word) 
        if word[-1] in EOS:
            if len(line) > 0:
                line.append('\n')
                sent = ' '.join(line)
                WriteFile(sent)
                line = []
    fp.close
if __name__ ==  "__main__":
    main()