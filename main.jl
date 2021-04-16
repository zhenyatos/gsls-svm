include("GSLS-SVM.jl")

add_noise = true
add_model_plot = true
add_RMSE_plot = true
noise_σ = 0.2
max_sv_num = 10
γ = 1e+4

function main()
    # Regressor and regressand
    X = LinRange(0, 5.0, 100)
    𝑿 = [[x] for x in X]
    𝒚 = [sinc.(x) for x in X]
    𝒚 += add_noise * rand(Normal(0, 0.2), length(𝒚))
    𝒚 = transpose(𝒚)

    # Trained SVR model
    function get_SVR_model(𝜷, b, dict_indices)
        return (x) -> b + sum([𝜷[i] * 𝒦(𝑿[dict_indices[i]], [x])
                                        for i=1:length(dict_indices)])
    end

    # Plotting model
    if add_model_plot
        dict_indices, 𝜷, b, det_H_vals = GSLS_SVM(𝑿, 𝒚, γ, 9)
        x = 0:0.01:5.0
        y1 = sinc.(x)
        f = get_SVR_model(𝜷, b, dict_indices)
        y2 = f.(x)
        plot(x, y1, label="theoretical", color="gray", dpi=300)
        plot!(x, y2, label="empirical", color="black")
        scatter!(LinRange(0, 5.0, 100), transpose(𝒚),
                                markersize=2,
                                markerstrokewidth=0.5,
                                label="samples",
                                color="pink")
        scatter!(LinRange(0, 5.0, 100)[dict_indices], transpose(𝒚)[dict_indices],
                                markersize=3,
                                label="support vectors",
                                color="red")
        savefig("model.png")
    end

    # Plotting dependency of RMSE from number of support vectors
    if add_RMSE_plot
        sv_nums = 1:max_sv_num
        RMSE_vals = []
        for n in sv_nums
            dict_indices, 𝜷, b, det_H_vals = GSLS_SVM(𝑿, 𝒚, γ, n)
            f = get_SVR_model(𝜷, b, dict_indices)
            y = f.(transpose(X))
            push!(RMSE_vals, RMSE(y, 𝒚))
        end
        plot(sv_nums, RMSE_vals, dpi=300, label="RMSE", xticks=1:max_sv_num)
        savefig("RMSE.png")
    end
end

main()
