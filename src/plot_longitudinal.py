import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class LongitudinalChangePlotter:
    def __init__(self, dataframe, subject_col="Subject"):
        self.df = dataframe
        self.subject_col = subject_col
        self.region_columns = [col for col in self.df.columns if col != subject_col]
    
    def preprocess(self):
        long_df = pd.melt(self.df, id_vars=[self.subject_col], value_vars=self.region_columns,
                          var_name="SessionRegion", value_name="Volume")
        long_df[['ImageType', 'Session', 'Region']] = long_df['SessionRegion'].str.extract(r'(T2W|ZSSR)_ses-(\d{2})_(.+)')
        long_df.dropna(subset=["Volume"], inplace=True)
        long_df["Session"] = long_df["Session"].astype(int)

        # Keep only subjects/regions with complete data across both sessions
        check_df = long_df.pivot_table(index=[self.subject_col, "Region"], columns=["ImageType", "Session"], values="Volume")
        check_df.columns = [f"{img}_{sess}" for img, sess in check_df.columns]
        check_df = check_df.dropna().reset_index()
        filtered_df = pd.merge(long_df, check_df[[self.subject_col, "Region"]], on=[self.subject_col, "Region"])

        pivot_df = filtered_df.pivot_table(index=[self.subject_col, "Region", "ImageType"], columns="Session", values="Volume").reset_index()
        pivot_df.columns.name = None
        pivot_df.rename(columns={1: "Session1", 2: "Session2"}, inplace=True)
        pivot_df["AbsPercentChange"] = (abs(pivot_df["Session2"] - pivot_df["Session1"]) / pivot_df["Session1"]) * 100

        t2w_df = pivot_df[pivot_df["ImageType"] == "T2W"]
        zssr_df = pivot_df[pivot_df["ImageType"] == "ZSSR"]
        merged_df = pd.merge(
            t2w_df[[self.subject_col, "Region", "AbsPercentChange"]],
            zssr_df[[self.subject_col, "Region", "AbsPercentChange"]],
            on=[self.subject_col, "Region"],
            suffixes=("_T2W", "_ZSSR")
        )

        plot_df = pd.melt(merged_df, id_vars=[self.subject_col, "Region"],
                          value_vars=["AbsPercentChange_T2W", "AbsPercentChange_ZSSR"],
                          var_name="ImageType", value_name="AbsPercentChange")
        plot_df["ImageType"] = plot_df["ImageType"].str.replace("AbsPercentChange_", "")
        self.plot_df = plot_df

    def plot_bar_graph(self):
        plt.figure(figsize=(14, 6))
        sns.barplot(data=self.plot_df, x="Region", y="AbsPercentChange", hue="ImageType", ci="sd")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Mean Absolute % Change")
        plt.title("Bar Graph of Mean Absolute % Volume Change\nT2W vs ZSSR")
        plt.grid(True)
        plt.tight_layout()
        plt.legend(title="Image Type")
        plt.show()


# Load and use
df = pd.read_excel("/Users/niyathigirish/Documents/JHU/Output graphs/Output_Combined.xlsx")
plotter = LongitudinalChangePlotter(df)
plotter.preprocess()
plotter.plot_bar_graph()