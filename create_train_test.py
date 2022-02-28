import random

from modules.utils import set_seed

def save_data(data, fpath):
    with open(fpath, 'w') as out:
        for instance in data:
            for token in instance :
                out.write(token)
            out.write("\n")


if __name__ == "__main__" :

    set_seed(26092020)
    
    fpath = "data/ecommerce.txt"

    file = open(fpath)
    lines = file.readlines()
    file.close()

    data = []
    instance = []

    for l in lines :
        if l[:-1] == "": # if it's empty
            data.append(instance)
            instance = []
        else :
            instance.append(l)

    random.shuffle(data)

    train_size = int(0.9 * len(data))

    train_fpath = "data/ecommerce-train.txt"
    test_fpath = "data/ecommerce-test.txt"

    save_data(data[:train_size], train_fpath)
    save_data(data[train_size:], test_fpath)



    
    


