using Plots
using JLD2
using Lux
using NPZ
using MLUtils

include("./recurrent/convlstm.jl")

model = load("./model_config.jld2")["model"]




function get_dataloaders()
    ds = npzread("mnist_test_seq.npy")::Array{UInt8, 4} / Float32(typemax(UInt8))
    ds = permutedims(ds, (3, 4, 1, 2))#[:, :, :, 1:1_000]
    ds_x = reshape(ds[:, :, 1:10, :], (64, 64, 1, 10, :))
    ds_y = ds[:, :, 11:20, :]
    @show size(ds_x)
    @show size(ds_y)

    (x_train, y_train), (x_val, y_val) = splitobs((ds_x, ds_y); at=0.8)

    return (
        DataLoader((x_train, y_train); batchsize=32),
        DataLoader((x_val, y_val); batchsize=32),
    )
end

_, loader = get_dataloaders()

x, y = first(loader)

config = load("./models/trained_model_30.jld2")
ps, st = config["ps_trained"], Lux.testmode(config["st_trained"]);

ŷ, st_ = model(x, ps, st);

idx = 1

data_to_plot = vcat(
    reshape(ŷ[:, :, :, idx], 64, :),
    reshape(y[:, :, :, idx], 64, :),
    reshape(x[:, :, 1, :, idx], 64, :),
)

heatmap(data_to_plot, size=(128*10, 128*3))


