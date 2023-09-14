from urllib.request import urlretrieve
import os
import pandas as pd
import numpy as np
import plotnine as p9
from math import sqrt

zip_url="https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.test.gz"
zip_file_name="zip.test.gz"
image_number=500

#########
# Part 1
#########
if not os.path.exists(zip_file_name):
   print("Downloading...")
   urlretrieve(zip_url, zip_file_name)

#########
# Part 2
#########
zip_df = pd.read_csv(zip_file_name,sep=" ",header=None)
zip_df.shape

intensity_vec=zip_df.iloc[image_number,1:]
# 16 x 16 image
n_pixels=int(sqrt(len(intensity_vec)))

one_image_df = pd.DataFrame({
   "intensity":intensity_vec,
   "column":np.tile(np.arange(n_pixels), n_pixels),
   "row":np.flip(np.repeat(np.arange(n_pixels), n_pixels))
})

gg = p9.ggplot()+\
   p9.geom_tile(
      p9.aes(
            x="column",
            y="row",
            fill="intensity",
      ),
      data = one_image_df
   )

gg = gg + p9.ggtitle(zip_df.iloc[image_number,0])

print(gg)

#########
# Part 3
#########
dataframes = []

for image in range(9):
   image_df = pd.DataFrame({
      "intensity":zip_df.iloc[image,1:],
      "column":np.tile(np.arange(n_pixels), n_pixels),
      "row":np.flip(np.repeat(np.arange(n_pixels), n_pixels)),
      "observation":image,
      "label":zip_df.iloc[image,0]
   })
   
   dataframes.append(image_df)

big_dataframe = pd.concat(dataframes)

big_plot = p9.ggplot()+\
   p9.geom_tile(
      p9.aes(
            x="column",
            y="row",
            fill="intensity",
      ),
      data = big_dataframe
   )+\
   p9.facet_wrap(["observation", "label"], labeller="label_both")

print(big_plot)