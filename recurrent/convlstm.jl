include("./peepholeconvlstm.jl")


struct ConvLSTM{E, D, S} <: Lux.AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder::E
    decoder::D
    steps::S
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
    steps,
) where {N}
    return ConvLSTM(
        Recurrence(ConvLSTMCell(k_x, k_h, in_dims => hidden_dims, peephole=true)),
        ConvLSTMCell(k_x, k_h, hidden_dims => out_dims, peephole=true),
        steps,
    )
end

# WHCTN
function (c::ConvLSTM)(x::AbstractArray{T, N}, ps::NamedTuple, st::NamedTuple) where {T, N}
    (y, carry), st_encoder = c.encoder(x, ps.encoder, st.encoder)
    x_last = selectdim(x, N-1, size(x, N-1))
    (ys, carry), st_decoder = c.decoder((x_last, carry), ps.decoder, st.decoder)
    out = reshape(ys, size(ys)[1:N-2]..., :)
    for _ in 2:c.steps
        (ys, carry), st_decoder = c.decoder((ys, carry), ps.decoder, st_decoder)
        out = cat(out, ys; dims=Val(N-2))
    end
    return out, merge(st, (encoder=st_encoder, decoder=st_decoder))
end
