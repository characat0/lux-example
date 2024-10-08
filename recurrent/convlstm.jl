include("./peepholeconvlstm.jl")
include("./stacked.jl")


struct ConvLSTM{Teacher, E, D, C} <: Lux.AbstractLuxContainerLayer{(:encoder, :decoder, :last_conv)}
    teacher::Teacher
    encoder::E
    decoder::D
    last_conv::C
    steps
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
    in_dims, 
    hidden_dims, 
    out_dims,
    steps,
    activation=σ,
) where {N}
    return ConvLSTM(
        True(),
        Recurrence(
            StackedConvLSTMCell(
                ConvLSTMCell(k_x, k_h, in_dims => hidden_dims, peephole=true, use_bias=false),
                ConvLSTMCell(k_x, k_h, hidden_dims => hidden_dims ÷ 2, peephole=true, use_bias=false),
                ConvLSTMCell(k_x, k_h, hidden_dims ÷ 2 => hidden_dims ÷ 2, peephole=true, use_bias=false),
            ),
        ),
        StackedConvLSTMCell(
            ConvLSTMCell(k_x, k_h, in_dims => hidden_dims, peephole=true, use_bias=false),
            ConvLSTMCell(k_x, k_h, hidden_dims => hidden_dims ÷ 2, peephole=true, use_bias=false),
            ConvLSTMCell(k_x, k_h, hidden_dims ÷ 2 => hidden_dims ÷ 2, peephole=true, use_bias=false),
            concatenate=True(),
        ),
        Conv(ntuple(Returns(1), N), hidden_dims * 2 => out_dims, activation, use_bias=false),
        steps
    )
end

function (c::ConvLSTM{False})(x::AbstractArray{T, N}, ps::NamedTuple, st::NamedTuple) where {T, N}
    (_, carry), st_encoder = c.encoder(x, ps.encoder, st.encoder)
    # y_in = glorot_uniform(Lux.replicate(st_encoder.rng), size(x)[1:N-2]..., size(x, N)) |> get_device(x)
    output = selectdim(x, N-1, size(x, N-1))
    (ys, carry), st_decoder = c.decoder((output, carry), ps.decoder, st.decoder)
    output, st_last_conv = c.last_conv(ys, ps.last_conv, st.last_conv)
    out = reshape(output, size(output)[1:N-2]..., :)
    for _ in 2:c.steps
        (ys, carry), st_decoder = c.decoder((output, carry), ps.decoder, st_decoder)
        output, st_last_conv = c.last_conv(ys, ps.last_conv, st_last_conv)
        out = cat(out, output; dims=Val(N-2))
    end
    return out, merge(st, (encoder=st_encoder, decoder=st_decoder, last_conv=st_last_conv))
end


# WHCTN
function (c::ConvLSTM{True})(x::AbstractArray{T, N}, ps::NamedTuple, st::NamedTuple) where {T, N}
    X = x
    (_, carry), st_encoder = c.encoder(X, ps.encoder, st.encoder)
    Xi = zeros32(size(x)[1:N-2]..., size(x, N)) |> get_device(x)
    (ys, carry), st_decoder = c.decoder((Xi, carry), ps.decoder, st.decoder)
    output, st_last_conv = c.last_conv(ys, ps.last_conv, st.last_conv)
    out = output
    for i in 1:c.steps-1
        # Xi = selectdim(x, N-1, N_end+i)
        (ys, carry), st_decoder = c.decoder((output, carry), ps.decoder, st_decoder)
        output, st_last_conv = c.last_conv(ys, ps.last_conv, st_last_conv)
        out = cat(out, output; dims=Val(N-2))
    end
    return out, merge(st, (encoder=st_encoder, decoder=st_decoder, last_conv=st_last_conv))
end
