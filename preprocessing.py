# import pandas as pd
import numpy as np
import pandas as pd
import os


os.chdir(r'.\datasets')

# dirs to use:
dirs = ["EN", "EN_XX", "EN_XX_EN"]
# get full path to dirs:
paths = [os.path.join('.\\', d) for d in dirs]

for path in paths:
    # create list of csv files:
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    csv_files = [os.path.join(path, file) for file in csv_files]

    for infile in csv_files:
        df = pd.read_csv(infile)

        # try to set df column 'keywords' to null:
        try:
            df.keywords = None
        except:
            print("no 'keywords' column in {}".format(infile))

        df.to_csv(infile, index=False)
        print(infile + ' done')
