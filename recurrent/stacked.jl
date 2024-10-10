using Lux

@concrete struct StackedConvLSTMCell <: AbstractRecurrentCell
    layers::Lux.Chain
end

function StackedConvLSTMCell(cells...)
    StackedConvLSTMCell(Chain(cells...))
end


Lux.initialstates(rng::AbstractRNG, lstm::StackedConvLSTMCell) = Lux.initialstates(rng, lstm.layers)

Lux.initialparameters(rng::AbstractRNG, lstm::StackedConvLSTMCell) = Lux.initialparameters(rng, lstm.layers)

(s::StackedConvLSTMCell)(x, ps, st::NamedTuple) = applystacked(s.layers.layers, x, ps, st)

@generated function applystacked(
    layers::NamedTuple{fields}, x::AbstractArray, ps, st::NamedTuple{fields}) where {fields}
    N = length(fields)
    x_symbols = vcat([:x], [gensym() for _ in 1:N])
    c_symbols  = [gensym() for _ in 1:N]
    st_symbols = [gensym() for _ in 1:N]
    calls = [:((($(x_symbols[i + 1]), $(c_symbols[i])), $(st_symbols[i])) = @inline Lux.apply(
                 layers.$(fields[i]), $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i])))
             for i in 1:N]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(return ($(x_symbols[N + 1]), ($(c_symbols...), )), st))
    return Expr(:block, calls...)
end


@generated function applystacked(
    layers::NamedTuple{fields}, inp::Tuple{AbstractArray, Any}, ps, st::NamedTuple{fields}) where {fields}
    N = length(fields)
    x_symbols = vcat([:(inp[1])], [gensym() for _ in 1:N])
    c_symbols  = [gensym() for _ in 1:N]
    st_symbols = [gensym() for _ in 1:N]

    calls = [:((($(x_symbols[i + 1]), $(c_symbols[i])), $(st_symbols[i])) = @inline Lux.apply(
                 layers.$(fields[i]), ($(x_symbols[i]), inp[2][$(i)]), ps.$(fields[i]), st.$(fields[i])))
             for i in 1:N]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(return ($(x_symbols[N + 1]), ($(c_symbols...), )), st))
    return Expr(:block, calls...)
end


Base.show(io::IO, lstm::StackedConvLSTMCell) = print(io, "$(lstm.layers)")

