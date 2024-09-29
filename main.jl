using Lux, Random, Optimisers, Zygote
using CUDA, LuxCUDA
using NPZ
using MLUtils
using Printf
using ProgressMeter
using MLFlowClient
using JLD2
using Statistics
using Plots

mkpath("./artifacts")


include("./recurrent/convlstm.jl")

mlf = MLFlow()
experiment = getorcreateexperiment(mlf, "lux-mnist")


const lossfn = MSELoss()
matches(y_pred, y_true) = sum((y_pred .> 0.5f0) .== (y_true .> 0.5f0))
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)

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

function plot_predictions(model, train_state, data, run_info, epoch)
    x, y = data
    ps_trained, st_trained = (train_state.parameters, train_state.states)
    ŷ, _ = model(x, ps_trained, Lux.testmode(st_trained))

    for idx in [1, 3, 7]
        data_to_plot = vcat(
            reshape(ŷ[:, :, :, idx], 64, :),
            reshape(y[:, :, :, idx], 64, :),
            reshape(x[:, :, 1, :, idx], 64, :),
        ) |> cpu_device()
        fig = heatmap(data_to_plot, size=(128*10, 128*3), clims=(0, 1))
        savefig(fig, "./artifacts/epoch_$(lpad(epoch, 2, '0'))_predictions_$(idx)_step.png")
        logartifact(mlf, run_info, "./artifacts/epoch_$(lpad(epoch, 2, '0'))_predictions_$(idx)_step.png")
    end
end


function objective(
    run_info;
    device_id=1,
    logging=true,
    k_x,
    k_h,
    hidden,
    seed,
    eta,
    rho,
    n_steps,
)
    dev = gpu_device(device_id, force_gpu_usage=true)
    train_loader, val_loader = get_dataloaders() .|> dev
    steps = [1, 3, 5, 10]

    model = ConvLSTM((k_x, k_x), (k_h, k_h), 1, hidden, 1, 10)
    @save "./artifacts/model_config.jld2" model
    logartifact(mlf, run_info, "./artifacts/model_config.jld2")
    rng = Xoshiro(seed)
    ps, st = Lux.setup(rng, model) |> dev
    logparam(mlf, run_info, Dict(
        "rand.algo" => "Xoshiro",
        "rand.seed" => seed,
        "opt.algo" => "AdamW",
        "opt.rho" => rho,
        "opt.eta" => eta,
        "model.kernel_hidden" => k_h,
        "model.kernel_input" => k_x,
        "model.hidden_dims" => hidden,
    ))

    train_state = Training.TrainState(model, ps, st, RMSProp(; eta, rho))
    @info "Starting train"
    for epoch in 1:n_steps
        ## Train the model
        progress = Progress(length(train_loader); desc="Training Epoch $(epoch)", enabled=logging)
        losses = Float32[]
        for (x, y) in train_loader
            (_, loss, _, train_state) = Training.single_train_step!(
                AutoZygote(), lossfn, (x, y), train_state
            )
            push!(losses, loss)
            next!(progress; showvalues = [("loss", loss)])
        end
        logmetric(mlf, run_info, "loss_train", mean(losses); step=epoch)
        losses = Float32[]
        accuracies = Float32[]
        ## Validate the model
        progress = Progress(length(val_loader); desc="Training Epoch $(epoch)", enabled=logging)
        st_ = Lux.testmode(train_state.states)
        loss_at = Dict{Int, Vector{Float32}}()
        acc_at = Dict{Int, Vector{Float32}}()
        for s in steps
            loss_at[s] = Float32[]
            acc_at[s] = Float32[]
        end
        for (x, y) in val_loader
            ŷ, st_ = model(x, train_state.parameters, st_)
            loss = lossfn(ŷ, y)
            acc = accuracy(ŷ, y)
            for s in steps
                push!(loss_at[s], lossfn(ŷ[:, :, s, :], y[:, :, s, :]))
                push!(acc_at[s], accuracy(ŷ[:, :, s, :], y[:, :, s, :]))
            end
            push!(losses, loss)
            push!(accuracies, acc)
            next!(progress; showvalues = [("loss", loss), ("acc", acc)])
        end
        logmetric(mlf, run_info, "loss_test", mean(losses); step=epoch)
        logmetric(mlf, run_info, "acc_test", mean(accuracies); step=epoch)
        for s in steps
            logmetric(mlf, run_info, "acc_test.$(s)", mean(acc_at[s]); step=epoch)
            logmetric(mlf, run_info, "loss_test.$(s)", mean(loss_at[s]); step=epoch)
        end

        ps_trained, st_trained = (train_state.parameters, train_state.states) |> cpu_device()
        @save "./artifacts/trained_weights_$(epoch).jld2" ps_trained st_trained
        logartifact(mlf, run_info, "./artifacts/trained_weights_$(epoch).jld2")

        if ((epoch - 1) % 4 == 0) || (epoch == n_steps)
            plot_predictions(model, train_state, first(val_loader), run_info, epoch)
        end
    end

    ps_trained, st_trained = (train_state.parameters, train_state.states) |> cpu_device()
    @save "./artifacts/trained_model.jld2" ps_trained st_trained
    logartifact(mlf, run_info, "./artifacts/trained_model.jld2")
end

function objective(; kwargs...)
    run_info = createrun(mlf, experiment)
    @show run_info.info.run_name
    try
        objective(run_info; kwargs...)
        updaterun(mlf, run_info, "FINISHED")
    catch e
        if typeof(e) <: InterruptException
            updaterun(mlf, run_info, "KILLED")
        else
            updaterun(mlf, run_info, "FAILED")
        end
        rethrow()
    end
end


objective(;
    k_h=5,
    k_x=5,
    hidden=128,
    seed=42,
    eta=0.001,
    rho=0.9,
    n_steps=20,
)
