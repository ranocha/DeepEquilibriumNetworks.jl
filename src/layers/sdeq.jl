struct SkipDeepEquilibriumNetwork{M,S,J,R,P,RE1,RE2,A,Se,K} <: AbstractDeepEquilibriumNetwork
    jacobian_regularization::Bool
    residual_regularization::Bool
    model::M
    shortcut::S
    p::P
    re1::RE1
    re2::RE2
    split_idx::Int
    args::A
    kwargs::K
    sensealg::Se
    stats::DEQTrainingStats

    function SkipDeepEquilibriumNetwork(jacobian_regularization, residual_regularization, model, shortcut, p, re1, re2,
                                        split_idx, args, kwargs, sensealg, stats)
        p1, re1 = destructure_parameters(model)
        split_idx = length(p1)
        p2, re2 = shortcut === nothing ? ([], nothing) : destructure_parameters(shortcut)

        p = p === nothing ? vcat(p1, p2) : convert(typeof(p1), p)

        return new{typeof(model),typeof(shortcut),jacobian_regularization,residual_regularization,typeof(p),typeof(re1),
                   typeof(re2),typeof(args),typeof(sensealg),typeof(kwargs)}(jacobian_regularization,
                                                                             residual_regularization, model, shortcut, p,
                                                                             re1, re2, split_idx, args, kwargs,
                                                                             sensealg, stats)
    end
end

Flux.@functor SkipDeepEquilibriumNetwork

function Base.show(io::IO, l::SkipDeepEquilibriumNetwork{M,S,J,R}) where {M,S,J,R}
    shortcut_ps = l.split_idx == length(l.p) ? 0 : length(l.p) - l.split_idx
    return print(io, "SkipDeepEquilibriumNetwork(jacobian_regularization = $J, residual_regularization = $R, ",
                 "shortcut_parameter_count = $shortcut_ps) ", string(length(l.p)), " Trainable Parameters")
end

function SkipDeepEquilibriumNetwork(model, shortcut, solver; p=nothing, jacobian_regularization::Bool=false,
                                    residual_regularization::Bool=false, sensealg=get_default_ssadjoint(0.1f0, 0.1f0, 10),
                                    kwargs...)
    return SkipDeepEquilibriumNetwork(jacobian_regularization, residual_regularization, model, shortcut, p, nothing,
                                      nothing, 0, (solver,), kwargs, sensealg, DEQTrainingStats(0))
end

function SkipDeepEquilibriumNetwork(model, solver; p=nothing, jacobian_regularization::Bool=false,
                                    residual_regularization::Bool=false, sensealg=get_default_ssadjoint(0.1f0, 0.1f0, 10),
                                    kwargs...)
    return SkipDeepEquilibriumNetwork(jacobian_regularization, residual_regularization, model, nothing, p, nothing,
                                      nothing, 0, (solver,), kwargs, sensealg, DEQTrainingStats(0))
end

function (deq::SkipDeepEquilibriumNetwork)(x::AbstractArray{T}) where {T}
    p1, p2 = deq.p[1:(deq.split_idx)], deq.p[(deq.split_idx + 1):end]
    z = deq.re2(p2)(x)::typeof(x)

    # Dummy call to ensure that mask is generated
    Zygote.@ignore _ = deq.re1(p1)(z, x)

    z_star = solve_steady_state_problem(deq.re1, p1, x, z, deq.sensealg, deq.args...; dudt=nothing,
                                        update_nfe=() -> (deq.stats.nfe += 1), deq.kwargs...)

    jac_loss = (deq.jacobian_regularization ? compute_deq_jacobian_loss(deq.re1, p1, z_star, x) : T(0)) ::T

    residual = if deq.residual_regularization
        z_star .- deq.re1(p1)(z_star, x)
    else
        Zygote.@ignore z_star .- deq.re1(p1)(z_star, x)
    end

    return z_star, DeepEquilibriumSolution(z_star, z, residual, jac_loss)
end

function (deq::SkipDeepEquilibriumNetwork{M,Nothing})(x::AbstractArray{T}) where {M,T}
    z = deq.re1(deq.p)(zero(x), x)::typeof(x)

    z_star = solve_steady_state_problem(deq.re1, deq.p, x, z, deq.sensealg, deq.args...; dudt=nothing,
                                        update_nfe=() -> (deq.stats.nfe += 1), deq.kwargs...)
    
    jac_loss = (deq.jacobian_regularization ? compute_deq_jacobian_loss(deq.re1, deq.p, z_star, x) : T(0)) ::T

    residual = if deq.residual_regularization
        z_star .- deq.re1(deq.p)(z_star, x)
    else
        Zygote.@ignore z_star .- deq.re1(deq.p)(z_star, x)
    end

    return z_star, DeepEquilibriumSolution(z_star, z, residual, jac_loss)
end

# For multiple argument functions
function (deq::SkipDeepEquilibriumNetwork)(x::AbstractMatrix{T}, x1::AbstractArray{T}, args...) where {T}
    p1, p2 = deq.p[1:(deq.split_idx)], deq.p[(deq.split_idx + 1):end]
    u0 = first(deq.re2(p2)(x, x1, args...))

    function dudt(u, _p, t)
        deq.stats.nfe += 1
        return first(deq.re1(_p)(u, x, x1, args...)) .- u
    end

    ssprob = SteadyStateProblem(dudt, u0, p1)
    sol = solve(ssprob, deq.args...; u0=u0, sensealg=deq.sensealg, deq.kwargs...)
    deq.stats.nfe += 1

    z_star = first(deq.re1(p1)(sol.u, x, x1, args...))

    jac_loss = T(0)

    residual = if deq.residual_regularization
        z_star .- first(deq.re1(p1)(z_star, x, x1, args...))
    else
        Zygote.@ignore z_star .- first(deq.re1(p1)(z_star, x, x1, args...))
    end

    return (z_star, x1, args...), DeepEquilibriumSolution(z_star, u0, residual, jac_loss)
end

function (deq::SkipDeepEquilibriumNetwork{M,Nothing})(x::AbstractMatrix{T}, x1::AbstractArray{T}, args...) where {M,T}
    u0 = first(deq.re1(deq.p)(x, x1, args...))

    function dudt(u, _p, t)
        deq.stats.nfe += 1
        return first(deq.re1(_p)(u, x, x1, args...)) .- u
    end

    ssprob = SteadyStateProblem(dudt, u0, deq.p)
    sol = solve(ssprob, deq.args...; u0=u0, sensealg=deq.sensealg, deq.kwargs...)
    deq.stats.nfe += 1

    z_star = first(deq.re1(deq.p)(lapl, sol.u, x))

    jac_loss = T(0)

    residual = if deq.residual_regularization
        z_star .- first(deq.re1(deq.p)(z_star, x, x1, args...))
    else
        Zygote.@ignore z_star .- first(deq.re1(deq.p)(z_star, x, x1, args...))
    end

    return (z_star, x1, args...), DeepEquilibriumSolution(z_star, u0, residual, jac_loss)
end
