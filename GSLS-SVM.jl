module MySVM

using LinearAlgebra

export kernel_RBF, kernel_polynomial, GSLS_SVM, RMSE

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
    return (𝒙, 𝒙′) -> (𝒙 ⋅ 𝒙′ + r)^n
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
    GSLS_SVM(𝒦, 𝑿, 𝒚, γ, sv_num, get_err_info=false)
Greedy Sparse Least-Squares SVM.

## Arguments
`𝒦` - kernel\
`𝑿` - dataset,\
`𝒚` - outcomes,\
`γ` - regularization parameter,\
`sv_num` - number of support vectors,\
`get_err_info` - set this to `true` if you want to get `err_vals` in the output.

## Output
`dict_indices` - support vectors indices,\
`best_𝜷` - 𝜷 of constructed SVM,\
`best_b` - b of constructed SVM,\
`err_vals` - list of error values to check that in each step solution of the system of
linear equations inside algorithm was correct.
    """
function GSLS_SVM(𝒦, 𝑿, 𝒚, γ, sv_num, get_err_info=false)
    ℓ = length(𝑿)
    dict_indices = []
    best_𝜷 = []
    best_b = 0
    best_inv_H = []
    best_index = 0
    err_vals = []

    for i = 1:sv_num
        best_ℒ = Inf
        if i != 1
            inv_H_old = copy(best_inv_H)
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
            local inv_H
            local rs

            H = [ℓ transpose(Φ); Φ Ω]
            if i == 1
                rs = [sum(𝒚); c]
                inv_H = inv(H)
            else
                m = size(H)[1]
                𝐚 = H[1:m-1, m]
                𝐛 = inv_H_old * 𝐚
                inv_k = 1 / (H[m, m] - dot(𝐚, 𝐛))
                A = inv_H_old + inv_k * 𝐛 * 𝐛'
                B = -inv_k * 𝐛
                inv_H = [A B; B' inv_k]
                rs = [sum(𝒚); c]
            end

            solution = inv_H * rs
            𝜷 = solution[2:i+1]
            b = solution[1]

            current_ℒ = ℒ(𝒦, 𝑿, 𝒚, 𝜷, b, dict_indices, γ)
            if current_ℒ < best_ℒ
                best_ℒ = current_ℒ
                best_𝜷 = copy(𝜷)
                best_b = b
                best_inv_H = copy(inv_H)
                best_index = j
            end
            pop!(dict_indices)
        end
        push!(dict_indices, best_index)

        if get_err_info
            Ω = 𝜴(𝒦, 𝑿, dict_indices, γ)
            Φ = 𝜱(𝒦, 𝑿, dict_indices)
            c = 𝒄(𝒦, 𝑿, 𝒚, dict_indices)
            err = norm([Ω Φ; transpose(Φ) ℓ] * [best_b; best_𝜷] - [sum(𝒚); c])
            push!(err_vals, err)
        end
    end

    if get_err_info
        return dict_indices, best_𝜷, best_b, err_vals
    else
        return dict_indices, best_𝜷, best_b
    end
end

end