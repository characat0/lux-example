using Lux, Random, Optimisers, Zygote
using Metal
using NPZ
using MLUtils
using Printf
using ProgressMeter
using JLD2

include("./recurrent/peepholeconvlstm.jl")

struct ConvLSTM{E, D, C} <: Lux.AbstractLuxContainerLayer{(:encoder, :decoder, :last_conv)}
    encoder::E
    decoder::D
    last_conv::C
end

function (r::Recurrence{False})(x::Union{AbstractVector, NTuple}, ps, st::NamedTuple)
    (out, carry), st = Lux.apply(r.cell, first(x), ps, st)
    for xᵢ in x[(begin + 1):end]
        (out, carry), st = Lux.apply(r.cell, (xᵢ, carry), ps, st)
    end
    return (out, carry), st
end

function ConvLSTM(
    k_x::NTuple{N},
    k_h::NTuple{N},
    in_dims, hidden_dims, out_dims,
) where {N}
    return ConvLSTM(
        Recurrence(ConvLSTMCell(k_x, k_h, in_dims => hidden_dims, peephole=false)),
        ConvLSTMCell(k_x, k_h, hidden_dims => hidden_dims, peephole=false),
        Conv(ntuple(Returns(1), N), hidden_dims => out_dims, use_bias=false),
    )
end

function (c::ConvLSTM)(x::AbstractArray{T, 5}, ps::NamedTuple, st::NamedTuple) where {T}
    (y, carry), st_encoder = c.encoder(x, ps.encoder, st.encoder)
    (ys, carry), st_decoder = c.decoder((y, carry), ps.decoder, st.decoder)
    output, st_last_conv = c.last_conv(ys, ps.last_conv, st.last_conv)
    out = reshape(output, 64, 64, 1, :)
    for _ in 2:10
        (ys, carry), st_decoder = c.decoder((ys, carry), ps.decoder, st_decoder)
        output, st_last_conv = c.last_conv(ys, ps.last_conv, st_last_conv)
        out = cat(out, output; dims=Val(3))
    end
    return out, merge(st, (encoder=st_encoder, decoder=st_decoder, last_conv=st_last_conv))
end


# function ConvLSTM()
#     encoder = Recurrence(ConvLSTMCell((5, 5), (5, 5), 1 => 8))
#     decoder = ConvLSTMCell((5, 5), (5, 5), 8 => 8)
#     conv_last = Conv((1, 1), 8 => 1, use_bias=false)

#     @compact(; encoder, decoder, conv_last) do x::AbstractArray{T, 5} where {T}
#         y = encoder(x)
#         @info "kho"
#         @show size(y)
#         ys, carry = decoder((y, carry))
#         out = [conv_last(ys)]
#         for _ in 2:10
#             ys, carry = decoder((y, carry))
#             out = vcat(out, [conv_last(ys)])
#         end
#         @return cat(out; dims=Val(3))
#     end
# end

const lossfn = BinaryCrossEntropyLoss(; logits=Val(true))
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

function main()

    dev = cpu_device()
    train_loader, val_loader = get_dataloaders() .|> dev
    model = ConvLSTM((5, 5), (5, 5), 1, 64, 1)
    rng = Xoshiro(42)
    ps, st = Lux.setup(rng, model) |> dev
    train_state = Training.TrainState(model, ps, st, Adam(0.01f0))
    @info "Starting train"
    for epoch in 1:3
        ## Train the model
        progress = Progress(length(train_loader); desc="Training Epoch $(epoch)")
        for (x, y) in train_loader
            (_, loss, _, train_state) = Training.single_train_step!(
                AutoZygote(), lossfn, (x, y), train_state
            )
            next!(progress; showvalues = [("loss", loss)])
        end

        ## Validate the model
        st_ = Lux.testmode(train_state.states)
        for (x, y) in val_loader
            ŷ, st_ = model(x, train_state.parameters, st_)
            loss = lossfn(ŷ, y)
            acc = accuracy(ŷ, y)
            @printf "Validation: Loss %4.5f Accuracy %4.5f\n" loss acc
        end
    end
    return (model, train_state.parameters, train_state.states) |> cpu_device()
end

model, ps_trained, st_trained = main()

@save "trained_model.jld2" ps_trained st_trained

@save "model_config.jld2" model
