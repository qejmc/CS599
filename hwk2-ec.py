import pandas as pd
import numpy as np
import plotnine as p9

data_set = "WorldBank.csv"
csv_df = pd.read_csv(data_set,sep=",",header=0)

WorldBank = csv_df
WorldBank1975 = csv_df[csv_df["year"] == 1975]
WorldBankBefore1975 = csv_df[(csv_df["year"] <= 1975) & 
                             (csv_df["year"] >= 1970)]

scatter = p9.ggplot()+\
   p9.geom_point(
         p9.aes(
               x="life.expectancy",
               y="fertility.rate",
               color="region",
         ),
         data = WorldBank1975
      )

viz_two_layers = scatter +\
   p9.geom_path(
         p9.aes(
               x="life.expectancy",
               y="fertility.rate",
               color="region",
               group="country",
         ),
         data = WorldBankBefore1975
      )

viz_two_plots = []
viz_two_plots.append(viz_two_layers)

viz_time_series = p9.ggplot()+\
   p9.geom_line(
         p9.aes(
               x="year",
               y="fertility.rate",
               color="region",
               group="country",
         ),
         data = WorldBank
      )

viz_two_plots.append(viz_time_series)

#########
# Part 1
#########
def add_x_var(df, x_var):
   df["x_var"] = pd.Categorical(df[x_var])
   return df

WorldBank_a = add_x_var(WorldBank.copy(), "year")
WorldBank1975_a = add_x_var(WorldBank1975.copy(), "life.expectancy")
WorldBankBefore1975_a = add_x_var(WorldBankBefore1975.copy(), "life.expectancy")

print(WorldBank1975_a.info())

viz_aligned = p9.ggplot()+\
   p9.geom_point(
         p9.aes(
               x="life.expectancy",
               y="fertility.rate",
               color="region",
         ),
         data = WorldBank1975_a
      )+\
   p9.geom_path(
         p9.aes(
               x="life.expectancy",
               y="fertility.rate",
               color="region",
               group="country",
         ),
         data = WorldBankBefore1975_a
      )+\
   p9.geom_line(
         p9.aes(
               x="year",
               y="fertility.rate",
               color="region",
               group="country",
         ),
         data = WorldBank_a
      )+\
   p9.xlab("")+\
   p9.facet_grid(". ~ x_var", scales="free")+\
   p9.theme_bw()+\
   p9.theme(panel_spacing=0)

print(viz_aligned)