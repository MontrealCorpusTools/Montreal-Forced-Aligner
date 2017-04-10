import sys

def compare(a, b):
    created = {}
    actual = {}
    with open(a) as f1:
        lines1 = f1.readlines()
        for line in lines1:
            splitline = line.split("\t")
            try:
                word, value = splitline[0], splitline[1]
            except IndexError:
                print(line)
            created.update({word:value})

    with open(b) as f2:
        lines2 = f2.readlines()
        for line in lines2:
            splitline = line.split("\t")
            try:
                word, value = splitline[0], splitline[1]
            except IndexError:
                print(line)
            actual.update({word:value})


    count_right = 0
    count_actual = 0
    actual_word = ""
    not_in_words = 0
    not_included  = []

    for k,v in created.items():
        try:
            actual_word = actual[k]
            count_actual +=1
        except KeyError:
            print("not included")
            not_in_words+=1
            not_included.append(k)
        if actual_word.strip() == v.strip():
            count_right += 1

    print(list(set(not_included))[0:100])

    print(count_right/count_actual, len(actual.keys()), not_in_words)
    return (count_right/count_actual, not_in_words)

        
