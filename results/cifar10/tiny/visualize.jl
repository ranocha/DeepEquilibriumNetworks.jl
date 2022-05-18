using Pkg
Pkg.activate(".")

using Plots, Pandas, Seaborn, Serialization, OrdinaryDiffEq

dirs = filter(isdir, readdir("."))

dfs = []

for dir in dirs
    setup = deserialize(joinpath(dir, "setup.jls"))
    config = setup["config"]
    args = setup["args"]
    df = Pandas.read_csv(joinpath(dir, "results.csv"))
    df["Seed"] = repeat([config.seed], size(df, 1))
    df["Model Type"] = repeat([config.model_type], size(df, 1))

    push!(dfs, df)
end

df = Pandas.concat(dfs)

Seaborn.seaborn.set(; font_scale=1.3)

df_melt = Pandas.melt(
    df;
    id_vars=["Epoch", "Model Type"],
    value_vars=[
        "Test/Accuracy",
        "Test/NFE",
        "Test/Batch Time",
        "Test/Residual",
        "Train/Accuracy",
        "Train/NFE",
        "Train/Batch Time",
        "Train/Residual",
    ],
    var_name="Metric Full",
    value_name="Value",
)
_splitted = split.(df_melt["Metric Full"], "/")

df_melt["Metric"] = last.(_splitted)
df_melt["Mode"] = first.(_splitted)

plt = Seaborn.seaborn.relplot(x="Epoch", y="Value", data=df_melt, hue="Model Type", col="Metric", row="Mode", facet_kws=Dict("sharey" => false, "legend_out" => false, "margin_titles" => true), kind="line");
Seaborn.plt.tight_layout()
plt.savefig("cifar10-tiny.pdf", bbox_inches="tight")