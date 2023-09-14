from urllib.request import urlretrieve
import os
import pandas

zip_url="https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.test.gz"
zip_file_name="zip.test.gz"
if not os.path.exists(zip_file_name):
    print("Downloading...")
    urlretrieve(zip_url, zip_file_name)

zip_df = pandas.read_csv(zip_file_name,sep=" ",header=None)
print(zip_df)
zip_df.shape
