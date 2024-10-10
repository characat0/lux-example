using Lux
using Lux: BoolType, False, True

@concrete struct StackedConvLSTMCell <: AbstractRecurrentCell
    concatenate <: StaticBool
    cells::Lux.Chain
end

function StackedConvLSTMCell(cells...; concatenate::BoolType=False())
    StackedConvLSTMCell(static(concatenate), Chain(cells...))
end


Lux.initialstates(rng::AbstractRNG, lstm::StackedConvLSTMCell) = Lux.initialstates(rng, lstm.cells)

Lux.initialparameters(rng::AbstractRNG, lstm::StackedConvLSTMCell) = Lux.initialparameters(rng, lstm.cells)

(s::StackedConvLSTMCell)(x, ps, st::NamedTuple) = applystacked(s.cells.layers, s.concatenate, x, ps, st)

@generated function applystacked(
    layers::NamedTuple{fields}, ::Lux.StaticBool{concat}, x::AbstractArray{T, ND}, ps, st::NamedTuple{fields}) where {fields, concat, ND, T}
    N = length(fields)
    x_symbols = vcat([:x], [gensym() for _ in 1:N])
    c_symbols  = [gensym() for _ in 1:N]
    st_symbols = [gensym() for _ in 1:N]
    calls = [:((($(x_symbols[i + 1]), $(c_symbols[i])), $(st_symbols[i])) = @inline Lux.apply(
                 layers.$(fields[i]), $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i])))
             for i in 1:N]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    if concat
        push!(calls, :(return (cat($(x_symbols[2:end]...); dims=ND-1), ($(c_symbols...), )), st))
    else
        push!(calls, :(return ($(x_symbols[N + 1]), ($(c_symbols...), )), st))
    end
    return Expr(:block, calls...)
end


@generated function applystacked(
    layers::NamedTuple{fields}, ::Lux.StaticBool{concat}, inp::Tuple{AbstractArray{T, ND}, Any}, ps, st::NamedTuple{fields}) where {fields, concat, ND, T}
    N = length(fields)
    x_symbols = vcat([:(inp[1])], [gensym() for _ in 1:N])
    c_symbols  = [gensym() for _ in 1:N]
    st_symbols = [gensym() for _ in 1:N]

    calls = [:((($(x_symbols[i + 1]), $(c_symbols[i])), $(st_symbols[i])) = @inline Lux.apply(
                 layers.$(fields[i]), ($(x_symbols[i]), inp[2][$(i)]), ps.$(fields[i]), st.$(fields[i])))
             for i in 1:N]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    if concat
        push!(calls, :(return (cat($(x_symbols[2:end]...); dims=ND-1), ($(c_symbols...), )), st))
    else
        push!(calls, :(return ($(x_symbols[N + 1]), ($(c_symbols...), )), st))
    end
    return Expr(:block, calls...)
end


Lux.Functors.children(x::StackedConvLSTMCell) = Lux.Functors.children(x.cells.layers)

function Base.show(io::IO, lstm::StackedConvLSTMCell)
    print(io, "StackedConvLSTMCell(\n")
    for (k, c) in pairs(Lux.Functors.children(lstm))
        Lux.PrettyPrinting.big_show(io, c, 4, k)
    end
    print(io, ")")
end
