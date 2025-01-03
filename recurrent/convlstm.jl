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
    hidden_dims::NTuple{M}, 
    out_dims,
    steps,
    use_bias::NTuple{M},
    peephole::NTuple{M},
    activation=σ,
) where {N, M}
    dims = vcat([in_dims], hidden_dims...)
    return ConvLSTM(
        True(),
        Recurrence(
            StackedConvLSTMCell(
                [
                    ConvLSTMCell(k_x, k_h, dims[i] => dims[i+1], peephole=peephole[i], use_bias=use_bias[i])
                    for i in 1:M
                ]...
            ),
        ),
        StackedConvLSTMCell(
            [
                ConvLSTMCell(k_x, k_h, dims[i] => dims[i+1], peephole=peephole[i], use_bias=use_bias[i])
                for i in 1:M
            ]...,
            concatenate=True(),
        ),
        Conv(ntuple(Returns(1), N), sum(hidden_dims) => out_dims, activation, use_bias=false),
        steps
    )
end

ConvLSTM(
    k_x::NTuple{N},
    k_h::NTuple{N},
    in_dims, 
    hidden_dims::NTuple{M}, 
    out_dims,
    steps,
    use_bias::Bool = false,
    peephole::Bool = true,
    activation=σ,
) where {N, M} = ConvLSTM(
    k_x,
    k_h,
    in_dims, 
    hidden_dims, 
    out_dims,
    steps,
    ntuple(Returns(use_bias), M),
    ntuple(Returns(peephole), M),
    activation,
)

function (c::ConvLSTM{False})(x::AbstractArray{T, N}, ps::NamedTuple, st::NamedTuple) where {T, N}
    (_, carry), st_encoder = c.encoder(x, ps.encoder, st.encoder)
    # Last frame
    Xi = selectdim(x, N-1, size(x, N-1) + 0)

    (ys, carry), st_decoder = c.decoder((Xi, carry), ps.decoder, st.decoder)
    Xi, st_last_conv = c.last_conv(ys, ps.last_conv, st.last_conv)
    out = Xi
    for _ in 2:c.steps
        # Autoregressive part
        (ys, carry), st_decoder = c.decoder((Xi, carry), ps.decoder, st_decoder)
        Xi, st_last_conv = c.last_conv(ys, ps.last_conv, st_last_conv)
        out = cat(out, Xi; dims=Val(N-2))
    end
    return out, merge(st, (encoder=st_encoder, decoder=st_decoder, last_conv=st_last_conv))
end


# WHCTN
function (c::ConvLSTM{True})(x::AbstractArray{T, N}, ps::NamedTuple, st::NamedTuple) where {T, N}
    N_end = size(x, N-1) - c.steps
    X = selectdim(x, N-1, 1:N_end)

    (_, carry), st_encoder = c.encoder(X, ps.encoder, st.encoder)
    # Last frame
    Xi = selectdim(x, N-1, size(X, N-1) + 0)

    (ys, carry), st_decoder = c.decoder((Xi, carry), ps.decoder, st.decoder)
    output, st_last_conv = c.last_conv(ys, ps.last_conv, st.last_conv)
    out = output
    for i in 1:c.steps-1
        # Teaching, we ignore the output of the last step, but use instead the correct frame
        Xi = selectdim(x, N-1, size(X, N-1) + i)
        # Autoregressive part
        (ys, carry), st_decoder = c.decoder((Xi, carry), ps.decoder, st_decoder)
        output, st_last_conv = c.last_conv(ys, ps.last_conv, st_last_conv)
        out = cat(out, output; dims=Val(N-2))
    end
    return out, merge(st, (encoder=st_encoder, decoder=st_decoder, last_conv=st_last_conv))
end
