using LinearAlgebra
using Plots
using Distributions

@doc raw"""
    kernel_RBF(σ)
Returns Radial Basis Function kernel
    ``𝒦 : ℝ^d × ℝ^d → ℝ``
defined as:\
    ``𝒦(𝒙, 𝒙^\prime) = \exp(- \frac{\| 𝒙 - 𝒙^\prime \|}{2σ^2})``
     """
function kernel_RBF(σ)
    if σ == 0
        throw(DomainError(σ, "Unsatisfied condition: σ ≠ 0"))
    end
    return (𝒙, 𝒙′) -> exp(-norm(𝒙 - 𝒙′)^2 / (2 * σ^2))
end

@doc raw"""
    kernel_polynomial(n, r)
Returns Polynomial kernel
    ``𝒦 : ℝ^d × ℝ^d → ℝ``
defined as:\
    ``𝒦(𝒙, 𝒙^\prime) = (𝒙 \cdot 𝒙^\prime + r)^n``
     """
function kernel_polynomial(n, r)
    if n < 1
        throw(DomainError(n, "Unsatisfied condition: n ⩾ 1"))
    end
    if r < 0
        throw(DomainError(r, "Unsatisfied condition: r ⩾ 0"))
    end
    return (𝒙, 𝒙′) -> (𝒙 ⋅ x′ + r)^n
end

@doc raw"""
    𝜴(𝑿, dict_indices)
Auxiliary function to build matrix
    ``\mathbf{\mathit{Ω}} ∈ ℝ^{n × n}``
where ``n`` is the size of a support vectors dictionary represented by
`dict_indices` and\
`𝒦` is a kernel,\
`𝑿` - dataset,\
`γ` - regularization parameter.
    """
function 𝜴(𝒦, 𝑿, dict_indices, γ)
    ℓ = length(𝑿)
    return [(ℓ / 2γ) * 𝒦(𝑿[i], 𝑿[j]) +
            sum([𝒦(𝑿[i], 𝑿[r]) * 𝒦(𝑿[r], 𝑿[j]) for r = 1:ℓ])
                for i in dict_indices, j in dict_indices]
end

@doc raw"""
    𝜱(𝑿, dict_indices)
Auxillary function to build column vector
    ``\mathbf{\mathit{Φ}} ∈ ℝ^{n × 1}``
where ``n`` is the size of a support vectors dictionary represented by
`dict_indices` and\
`𝒦` is a kernel,\
`𝑿` - dataset.
    """
function 𝜱(𝒦, 𝑿, dict_indices)
    return [sum(map((𝒙) -> 𝒦(𝑿[i], 𝒙), 𝑿)) for i in dict_indices]
end

@doc raw"""
    𝒄(𝑿, dict_indices)
Auxillary function to build column vector
    ``\mathbf{\mathit{c}} ∈ ℝ^{n × 1}``
where ``n`` is the size of a support vectors dictionary represented by
`dict_indices` and\
`𝒦` is a kernel\
`𝑿` - dataset.
    """
function 𝒄(𝒦, 𝑿, 𝒚, dict_indices)
    return [dot(𝒚, map((𝒙) -> 𝒦(𝑿[i], 𝒙), 𝑿)) for i in dict_indices]
end

@doc raw"""
    ℒ(𝒦, 𝑿, 𝒚, 𝜷, b, dict_indices)
Objective function for GSLS SVM
    ``ℒ : ℝ^n × ℝ → ℝ``
where ``n`` is the size of support vectors dictionary represented by
`dict_indices` and\
`𝒦` is a kernel\
`𝑿` - dataset, list of vectors,\
`𝒚` - outcomes (for the elements of 𝑿),\
`𝜷`, `b` - SVM coefficients,\
`γ` - regularization parameter.
    """
function ℒ(𝒦, 𝑿, 𝒚, 𝜷, b, dict_indices, γ)
    dict_length = length(dict_indices)
    sum1 = 0.5sum([𝜷[i] * 𝜷[j] * 𝒦(𝑿[dict_indices[i]], 𝑿[dict_indices[j]])
                for i = 1:dict_length, j = 1:dict_length])

    ℓ = length(𝑿)

    sum2 = 0
    for i = 1:ℓ
        sum3 = sum([𝜷[j] * 𝒦(𝑿[i], 𝑿[dict_indices[j]]) for j = 1:dict_length])
        sum2 += (𝒚[i] - sum3 - b)^2
    end
    sum2 *= (γ / ℓ)
    return sum1 + sum2
end

@doc """
    RMSE(y1, y2)
Root-mean-square error. Use this to check SVM learning quality: `y1` - predicted
outcomes, `y2` - true outcomes.
    """
function RMSE(y1, y2)
    return sqrt(sum((y1 - y2).^2) / length(y1))
end

@doc raw"""
    GSLS_SVM(𝒦, 𝑿, 𝒚, γ, sv_num)
Greedy Sparse Least-Squares SVM.

## Arguments
`𝒦` - kernel\
`𝑿` - dataset,\
`𝒚` - outcomes,\
`γ` - regularization parameter,\
`sv_num` - number of support vectors.

## Output
`dict_indices` - support vectors indices,\
`best_𝜷` - 𝜷 of constructed SVM,\
`best_b` - b of constructed SVM,\
`det_H_vals` - list of determinants of matrix H, which is constructed on each
step of the algorithm to find 𝜷 and b. If ``\det H ≈ 0`` than 𝜷 and b are probably
incorrect.
    """
function GSLS_SVM(𝒦, 𝑿, 𝒚, γ, sv_num)
    ℓ = length(𝑿)
    dict_indices = []
    best_𝜷 = []
    best_b = 0
    best_inv_Ω = []
    best_index = 0
    det_H_vals = Float64[]

    for i = 1:sv_num
        best_ℒ = Inf
        if i != 1
            inv_Ω_old = copy(best_inv_Ω)
        end
        for j = 1:ℓ
            if j in dict_indices
                continue
            end
            push!(dict_indices, j)
            Ω = 𝜴(𝒦, 𝑿, dict_indices, γ)
            Φ = 𝜱(𝒦, 𝑿, dict_indices)
            c = 𝒄(𝒦, 𝑿, 𝒚, dict_indices)

            local 𝜷
            local b
            local inv_Ω
            local solution
            if i == 1
                H = [Ω Φ; transpose(Φ) ℓ]
                rs = [c; sum(𝒚)]
                solution = H \ rs
                inv_Ω = (1 / Ω[1, 1]) * ones(1, 1)
            else
                m = size(Ω)[1]
                𝐛 = Ω[1:m-1, m]
                inv_k = 1 / (Ω[m, m] - transpose(𝐛) * inv_Ω_old * 𝐛)
                A = inv_Ω_old + inv_k * inv_Ω_old * 𝐛 * transpose(𝐛) * inv_Ω_old
                B = -inv_k * inv_Ω_old * 𝐛
                inv_Ω = [A B; transpose(B) inv_k]

                inv_k = 1 / (ℓ - transpose(Φ) * inv_Ω * Φ)
                A = inv_Ω + inv_k * inv_Ω * Φ * transpose(Φ) * inv_Ω
                B = -inv_k * inv_Ω * Φ
                inv_H = [A B; transpose(B) inv_k]

                rs = [c; sum(𝒚)]
                solution = inv_H * rs
            end

            𝜷 = solution[1:i]
            b = solution[i+1]
            current_ℒ = ℒ(𝒦, 𝑿, 𝒚, 𝜷, b, dict_indices, γ)
            if current_ℒ < best_ℒ
                best_ℒ = current_ℒ
                best_𝜷 = copy(𝜷)
                best_b = b
                best_inv_Ω = copy(inv_Ω)
                best_index = j
            end
            pop!(dict_indices)
        end
        push!(dict_indices, best_index)

        Ω = 𝜴(𝒦, 𝑿, dict_indices, γ)
        Φ = 𝜱(𝒦, 𝑿, dict_indices)
        push!(det_H_vals, det([Ω Φ; transpose(Φ) ℓ]))
    end

    return dict_indices, best_𝜷, best_b, det_H_vals
end
