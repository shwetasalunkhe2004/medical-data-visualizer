import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Create overweight column
df["overweight"] = (
    df["weight"] / ((df["height"] / 100) ** 2) > 25
).astype(int)

# Normalize cholesterol
df["cholesterol"] = (df["cholesterol"] > 1).astype(int)

# Normalize glucose
df["gluc"] = (df["gluc"] > 1).astype(int)


# --------------------------------------------------
# Categorical Plot
# --------------------------------------------------

def draw_cat_plot():

    # Create DataFrame for categorical plot
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=[
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "overweight"
        ]
    )

    # Group and count
    df_cat = (
        df_cat
        .groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # Create categorical plot
    cat_grid = sns.catplot(
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        data=df_cat,
        kind="bar"
    )

    cat_grid.set_ylabels("total")

    # Get actual Matplotlib Figure
    fig = cat_grid.fig

    # Save plot
    fig.savefig("catplot.png")

    return fig


# --------------------------------------------------
# Heat Map
# --------------------------------------------------

def draw_heat_map():

    # Clean data
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"]) &
        (df["height"] >= df["height"].quantile(0.025)) &
        (df["height"] <= df["height"].quantile(0.975)) &
        (df["weight"] >= df["weight"].quantile(0.025)) &
        (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # Calculate correlation
    corr = df_heat.corr()

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    # Save heatmap
    fig.savefig("heatmap.png")

    return fig