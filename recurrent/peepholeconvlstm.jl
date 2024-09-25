using Lux: AbstractRecurrentCell, StaticBool, True, False
using Lux: static
using Lux: BoolType
using Lux: IntegerType
using Lux: @concrete
using Lux
using Random

_ConvLSTMInputType = Tuple{<:AbstractArray, Tuple{<:AbstractArray, <:AbstractArray}}

@concrete struct ConvLSTMCell <: AbstractRecurrentCell
    train_state <: StaticBool
    train_memory <: StaticBool
    peephole <: StaticBool
    Wx
    Wh
    init_state
    init_memory
end

function ConvLSTMCell(
    k_i2h::NTuple{N,Integer},
    k_h2h::NTuple{N,Integer},
    (in_chs, out_chs)::Pair{<:Integer,<:Integer};
    use_bias::BoolType=True(),
    train_state::BoolType=False(), 
    train_memory::BoolType=False(),
    peephole::BoolType=True(),
    init_weight=nothing, 
    init_bias=nothing, 
    init_state=zeros32,
    init_memory=zeros32,
    ) where {N}
    input_to_hidden = in_chs => out_chs
    hidden_to_hidden = out_chs => out_chs

    Wx = Conv(k_i2h, input_to_hidden; init_weight, init_bias, pad=SamePad(), use_bias)
    Wh = Conv(k_h2h, hidden_to_hidden; init_weight, init_bias, pad=SamePad(), use_bias=False())

    return ConvLSTMCell(
        static(train_state), static(train_memory), static(peephole), Wx, Wh, init_state, init_memory,
    )
end

Lux.initialstates(rng::AbstractRNG, ::ConvLSTMCell) = (rng=Lux.Utils.sample_replicate(rng),)


function Lux.initialparameters(rng::AbstractRNG, lstm::ConvLSTMCell)
    ps = NamedTuple()
    for gate in ("i", "o", "f", "c")
        Wx = Lux.initialparameters(rng, lstm.Wx)
        Wh = Lux.initialparameters(rng, lstm.Wh)
        ps = merge(ps, NamedTuple([(Symbol("Wx_$(gate)"), Wx), (Symbol("Wh_$(gate)"), Wh)]))
    end
    if Lux.has_bias(lstm)
        bias_ih = vcat([init_rnn_bias(rng, init_bias, lstm.out_dims, lstm.out_dims)
                        for init_bias in lstm.init_bias]...)
        bias_hh = vcat([init_rnn_bias(rng, init_bias, lstm.out_dims, lstm.out_dims)
                        for init_bias in lstm.init_bias]...)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    Lux.has_train_state(lstm) &&
        (ps = merge(ps, (hidden_state=lstm.init_state(rng, lstm.out_dims),)))
    Lux.known(lstm.train_memory) &&
        (ps = merge(ps, (memory=lstm.init_memory(rng, lstm.out_dims),)))
    if Lux.known(lstm.peephole)
        shape = (ntuple(Returns(1), length(lstm.Wh.kernel_size))..., lstm.Wh.out_chs)
        ps = merge(ps, (
            Wc_i=Lux.init_rnn_weight(rng, nothing, lstm.Wh.out_chs, shape),
            Wc_f=Lux.init_rnn_weight(rng, nothing, lstm.Wh.out_chs, shape),
            Wc_o=Lux.init_rnn_weight(rng, nothing, lstm.Wh.out_chs, shape),
        ))
    end
    return ps
end

function calc_out_dims(W::NTuple{M}, padding::NTuple{N}, K::NTuple{M}, stride) where {N,M}
    if N == M 
      pad = padding .* 2
    elseif N==M*2
      pad = ntuple(i -> padding[2i-1] + padding[2i], M)
    end
    ((W .+ pad .- K) .÷ stride) .+ 1
end

function Lux.init_rnn_hidden_state(rng::AbstractRNG, lstm::ConvLSTMCell, x::AbstractArray{T, N}) where {T, N}
    # TODO: Once we support moving `rng` to the device, we can directly initialize on the
    #       device
    input_size = ntuple(i -> size(x, i), N-2)
    hidden_size = calc_out_dims(input_size, lstm.Wx.pad, lstm.Wx.kernel_size, lstm.Wx.stride)
    channels = lstm.Wh.in_chs
    lstm.init_state(rng, hidden_size..., channels, size(x, N))
end


function (lstm::ConvLSTMCell{False, False})(x::AbstractArray, ps, st::NamedTuple)
    rng = Lux.replicate(st.rng)
    hidden_state = Lux.init_rnn_hidden_state(rng, lstm, x)
    memory = Lux.init_rnn_hidden_state(rng, lstm, x)
    return lstm((x, (hidden_state, memory)), ps, merge(st, (; rng)))
end

function (lstm::ConvLSTMCell)((x, (hiddenₙ, memoryₙ))::_ConvLSTMInputType, ps, st::NamedTuple)
    Conv_xi₂, st_i = lstm.Wx(x, ps.Wx_i, st)
    st = merge(st, st_i)
    
    Conv_xf₂, st_i = lstm.Wx(x, ps.Wx_f, st)
    st = merge(st, st_i)

    Conv_xc₂, st_i = lstm.Wx(x, ps.Wx_c, st)
    st = merge(st, st_i)

    Conv_xo₂, st_i = lstm.Wx(x, ps.Wx_o, st)
    st = merge(st, st_i)


    Conv_hiₙ, st_h = lstm.Wh(hiddenₙ, ps.Wh_i, st)
    st = merge(st, st_h)
    
    Conv_hfₙ, st_h = lstm.Wh(hiddenₙ, ps.Wh_f, st)
    st = merge(st, st_h)

    Conv_hcₙ, st_h = lstm.Wh(hiddenₙ, ps.Wh_c, st)
    st = merge(st, st_h)

    Conv_hoₙ, st_h = lstm.Wh(hiddenₙ, ps.Wh_o, st)
    st = merge(st, st_h)

    if Lux.known(lstm.peephole)
        input₂  = @. sigmoid_fast(Conv_xi₂ + Conv_hiₙ + ps.Wc_i * memoryₙ)
        forget₂ = @. sigmoid_fast(Conv_xf₂ + Conv_hfₙ + ps.Wc_f * memoryₙ)
    else
        input₂  = @. sigmoid_fast(Conv_xi₂ + Conv_hiₙ)
        forget₂ = @. sigmoid_fast(Conv_xf₂ + Conv_hfₙ)
    end

    memory₂ = @. forget₂ * memoryₙ + input₂ * tanh_fast(Conv_xc₂ .+ Conv_hcₙ)

    if Lux.known(lstm.peephole)
        output₂ = @. sigmoid_fast(Conv_xo₂ + Conv_hoₙ + ps.Wc_o * memory₂)
    else
        output₂ = @. sigmoid_fast(Conv_xo₂ + Conv_hoₙ)
    end

    hidden₂ = @. output₂ * tanh_fast(memory₂)
    return (hidden₂, (hidden₂, memory₂)), st
end

function Base.show(io::IO, lstm::ConvLSTMCell)
    print(io, "ConvLSTMCell($(lstm.Wx.kernel_size), $(lstm.Wh.kernel_size), $(lstm.Wx.in_chs => lstm.Wx.out_chs)")
    all(==(0), lstm.Wx.pad) || print(io, ", pad=", Lux.PrettyPrinting.tuple_string(lstm.Wx.pad))
    Lux.has_bias(lstm.Wx) || print(io, ", use_bias=false")
    Lux.has_train_state(lstm) && print(io, ", train_state=true")
    Lux.known(lstm.train_memory) && print(io, ", train_memory=true")
    print(io, ")")
end


